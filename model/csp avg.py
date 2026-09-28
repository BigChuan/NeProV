import torch
from torch import nn

class CSPInterface(nn.Module):
    def __init__(self, tokenize, clip_model, dset,args):
        super().__init__()
        self.tokenize = tokenize
        self.config = args
        self.dset = dset
        self.clip_model = clip_model
        self.attr_dropout = nn.Dropout(getattr(args, "attr_dropout", args.attr_dropout))
        self.neg_attr_dropout = nn.Dropout(getattr(args, "neg_attr_dropout", args.attr_dropout))
        self.token_ids, soft_att_obj, neg_soft_att_obj, com_ctx_vectors, att_ctx_vectors, obj_ctx_vectors = self.construct_soft_prompt()
        self.com_ctx_vectors = nn.Parameter(com_ctx_vectors)
        self.att_ctx_vectors = nn.Parameter(att_ctx_vectors)
        self.obj_ctx_vectors = nn.Parameter(obj_ctx_vectors)
        self.soft_att_obj = nn.Parameter(soft_att_obj)
        self.neg_soft_att_obj = nn.Parameter(neg_soft_att_obj)

    def construct_soft_prompt(self):
        classes = [cla.replace(".", " ").lower() for cla in self.dset.objs]
        attributes = [attr.replace(".", " ").lower() for attr in self.dset.attrs]
        
        self.num_att = len(attributes)
        self.num_cls = len(classes)

        ctx_init = self.config.ctx_init
        assert isinstance(ctx_init, list)
        n_ctx = [len(ctx.split()) for ctx in ctx_init]
        prompt = self.tokenize(
            ctx_init,
            context_length=self.config.context_length,
        ).cuda()
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(prompt)

        comp_ctx_vectors = embedding[0, 1 : 1 + n_ctx[0], :].to(self.clip_model.dtype)
        attr_ctx_vectors = embedding[1, 1 : 1 + n_ctx[1], :].to(self.clip_model.dtype)
        obj_ctx_vectors = embedding[2, 1 : 1 + n_ctx[2], :].to(self.clip_model.dtype)
        
        token_ids = self.tokenize(
            self.config.prompt_template,
            context_length=self.config.context_length,
        ).cuda()

        tokenized = torch.cat(
            [
                self.tokenize(tok, context_length=self.config.context_length)
                for tok in attributes + classes
            ]
        )
        orig_token_embedding = self.clip_model.token_embedding(tokenized.cuda())
        soft_att_obj = torch.zeros(
            (len(attributes) + len(classes), orig_token_embedding.size(-1)),
            device=orig_token_embedding.device,
            dtype=orig_token_embedding.dtype,
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)


        Na = len(attributes)
        No = len(classes)

        attr_pos = soft_att_obj[:Na, :]
        obj_pos = soft_att_obj[Na:, :]

        if Na > 1:
            attr_sum = attr_pos.sum(dim=0, keepdim=True)
            attr_neg = (attr_sum - attr_pos) / (Na - 1)
        else:
            attr_neg = attr_pos.clone()

        if No > 1:
            obj_sum = obj_pos.sum(dim=0, keepdim=True)
            obj_neg = (obj_sum - obj_pos) / (No - 1)
        else:
            obj_neg = obj_pos.clone()

        neg_soft_att_obj = torch.cat([attr_neg, obj_neg], dim=0).to(dtype=soft_att_obj.dtype)

        return token_ids, soft_att_obj, neg_soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors


    def construct_token_tensors(self, pair_idx):
        """
        根据 pair_idx 构造正样本和多种负样本的 token embedding。

        返回一个 dict，包含：
            - 4 个 comp token:
                "comp_pos", "comp_neg1", "comp_neg2", "comp_neg3"
            - 2 个 attr token:
                "attr_pos", "attr_neg"
            - 2 个 obj token:
                "obj_pos", "obj_neg"

        各自形状：
            comp_*: [len(pair_idx), L, D]
            attr_*: [num_att, L, D]
            obj_* : [num_cls, L, D]
        """
        device = self.clip_model.token_embedding.weight.device
        dtype = self.clip_model.dtype

        # pair_idx -> (attr_idx, obj_idx)
        pair_idx = pair_idx.to(device)
        attr_idx = pair_idx[:, 0].long()  # [B]
        obj_idx  = pair_idx[:, 1].long()  # [B]
        B = pair_idx.size(0)

        # ------------------ 1. 准备三类 prompt 的 token ids ------------------ #
        # token_ids: [3, L]，依次为 comp / attr / obj
        comp_ids = self.token_ids[0].unsqueeze(0).repeat(B, 1).to(device)              # [B, L]
        attr_ids = self.token_ids[1].unsqueeze(0).repeat(self.num_att, 1).to(device)   # [num_att, L]
        obj_ids  = self.token_ids[2].unsqueeze(0).repeat(self.num_cls, 1).to(device)   # [num_cls, L]

        # 一次 embedding，后面用 clone 拆出不同分支
        comp_base = self.clip_model.token_embedding(comp_ids).type(dtype)  # [B, L, D]
        attr_base = self.clip_model.token_embedding(attr_ids).type(dtype)  # [num_att, L, D]
        obj_base  = self.clip_model.token_embedding(obj_ids).type(dtype)   # [num_cls, L, D]

        # ------------------ 2. 为各分支克隆底座 ------------------ #
        # comp：4 个版本
        comp_pos  = comp_base.clone()
        comp_neg1 = comp_base.clone()
        comp_neg2 = comp_base.clone()
        comp_neg3 = comp_base.clone()

        # attr：正 / 负
        attr_pos = attr_base.clone()
        attr_neg = attr_base.clone()

        # obj：正 / 负
        obj_pos = obj_base.clone()
        obj_neg = obj_base.clone()

        # ------------------ 3. 计算 </eos> 的位置 ------------------ #
        eos_comp = int(self.token_ids[0].argmax())
        eos_attr = int(self.token_ids[1].argmax())
        eos_obj  = int(self.token_ids[2].argmax())

        # ------------------ 4. 正 / 负 prototype ------------------ #
        soft_att_obj_pos = self.attr_dropout(self.soft_att_obj)        # [num_att + num_cls, D]
        soft_att_obj_neg = self.neg_attr_dropout(self.neg_soft_att_obj)    # [num_att + num_cls, D]

        # ------------------ 5. 填 comp 分支 ------------------ #
        # comp_pos:   attr 正 + obj 正
        comp_pos[:, eos_comp - 2, :] = soft_att_obj_pos[attr_idx].type(dtype)
        comp_pos[:, eos_comp - 1, :] = soft_att_obj_pos[obj_idx + self.num_att].type(dtype)
        comp_pos[:, 1:1 + len(self.com_ctx_vectors), :] = self.com_ctx_vectors.type(dtype)

        # comp_neg1:  attr 正 + obj 负
        comp_neg1[:, eos_comp - 2, :] = soft_att_obj_pos[attr_idx].type(dtype)
        comp_neg1[:, eos_comp - 1, :] = soft_att_obj_neg[obj_idx + self.num_att].type(dtype)
        comp_neg1[:, 1:1 + len(self.com_ctx_vectors), :] = self.com_ctx_vectors.type(dtype)

        # comp_neg2:  attr 负 + obj 正
        comp_neg2[:, eos_comp - 2, :] = soft_att_obj_neg[attr_idx].type(dtype)
        comp_neg2[:, eos_comp - 1, :] = soft_att_obj_pos[obj_idx + self.num_att].type(dtype)
        comp_neg2[:, 1:1 + len(self.com_ctx_vectors), :] = self.com_ctx_vectors.type(dtype)

        # comp_neg3:  attr 负 + obj 负
        comp_neg3[:, eos_comp - 2, :] = soft_att_obj_neg[attr_idx].type(dtype)
        comp_neg3[:, eos_comp - 1, :] = soft_att_obj_neg[obj_idx + self.num_att].type(dtype)
        comp_neg3[:, 1:1 + len(self.com_ctx_vectors), :] = self.com_ctx_vectors.type(dtype)

        # ------------------ 6. 填 attr 分支 ------------------ #
        # attr_pos: 所有属性用正 prototype
        attr_pos[:, eos_attr - 1, :] = soft_att_obj_pos[:self.num_att].type(dtype)
        attr_pos[:, 1:1 + len(self.att_ctx_vectors), :] = self.att_ctx_vectors.type(dtype)

        # attr_neg: 所有属性用负 prototype
        attr_neg[:, eos_attr - 1, :] = soft_att_obj_neg[:self.num_att].type(dtype)
        attr_neg[:, 1:1 + len(self.att_ctx_vectors), :] = self.att_ctx_vectors.type(dtype)

        # ------------------ 7. 填 obj 分支 ------------------ #
        # obj_pos: 所有类别用正 prototype
        obj_pos[:, eos_obj - 1, :] = soft_att_obj_pos[self.num_att:].type(dtype)
        obj_pos[:, 1:1 + len(self.obj_ctx_vectors), :] = self.obj_ctx_vectors.type(dtype)

        # obj_neg: 所有类别用负 prototype
        obj_neg[:, eos_obj - 1, :] = soft_att_obj_neg[self.num_att:].type(dtype)
        obj_neg[:, 1:1 + len(self.obj_ctx_vectors), :] = self.obj_ctx_vectors.type(dtype)

        # ------------------ 8. 汇总成一个 dict ------------------ #
        token_dict = {
            # 4 × comp
            "comp_pos":  comp_pos,
            "comp_neg_obj": comp_neg1,
            "comp_neg_att": comp_neg2,
            "comp_neg_both": comp_neg3,

            # 2 × attr
            "attr_pos": attr_pos,
            "attr_neg": attr_neg,

            # 2 × obj
            "obj_pos": obj_pos,
            "obj_neg": obj_neg,
        }

        return token_dict

    
