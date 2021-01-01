import transformers
import torch
from copy import deepcopy

from jiant.ext.radam import RAdam
from jiant.shared.model_resolution import ModelArchitectures, resolve_tokenizer_class


def get_tokenizer(model_type, tokenizer_path):
    """Instantiate a tokenizer for a given model type.

    Args:
        model_type (str): model shortcut name.
        tokenizer_path (str): path to tokenizer directory.

    Returns:
        Tokenizer for the given model type.

    """
    model_arch = ModelArchitectures.from_model_type(model_type)
    tokenizer_class = resolve_tokenizer_class(model_type)
    if model_arch in [ModelArchitectures.BERT]:
        if "-cased" in model_type:
            do_lower_case = False
        elif "-uncased" in model_type:
            do_lower_case = True
        else:
            raise RuntimeError(model_type)
    elif model_arch in [
        ModelArchitectures.XLM,
        ModelArchitectures.ROBERTA,
        ModelArchitectures.XLM_ROBERTA,
        ModelArchitectures.BART,
        ModelArchitectures.MBART,
        ModelArchitectures.ELECTRA,
    ]:
        do_lower_case = False
    elif model_arch in [ModelArchitectures.ALBERT]:
        do_lower_case = True
    else:
        raise RuntimeError(str(tokenizer_class))
    tokenizer = tokenizer_class.from_pretrained(tokenizer_path, do_lower_case=do_lower_case)
    return tokenizer


class OptimizerScheduler:
    def __init__(self, optimizer, scheduler):
        super().__init__()
        self.optimizer = optimizer
        self.scheduler = scheduler

    def step(self, skip_scheduler=False):
        self.optimizer.step()
        if not skip_scheduler:
            self.scheduler.step()

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict, strict=True):
        self.optimizer.load_state_dict(state_dict["optimizer"], strict=strict)
        self.scheduler.load_state_dict(state_dict["scheduler"], strict=strict)


class OptimizerSchedulerWithGradOps(OptimizerScheduler):
    def __init__(self, grad_sim_metric, grad_sim_nonlinear, **kwargs):
        super().__init__(**kwargs)
        self.grad_sim_metric = grad_sim_metric
        self.grad_sim_nonlinear = grad_sim_nonlinear
        if grad_sim_nonlinear == "":
            self.nonlinear_fn = lambda x: x
        elif grad_sim_nonlinear.startswith("stepfn"):
            self.threshold = float(self.grad_sim_nonlinear.split("_")[1])
            self.nonlinear_fn = lambda x: (x > self.threshold).float()
        elif grad_sim_nonlinear == "relu":
            self.nonlinear_fn = torch.relu
        elif grad_sim_nonlinear == "sqr":
            self.nonlinear_fn = lambda x: (x * x)
        # self.count = 0

    def get_shared_grad(self, copy=False, get_base=True):
        shared_param_grad = [
            [p.grad for p, is_base in zip(g["params"], g["is_base_encoder"]) if get_base == is_base]
            if g["shared"]
            else []
            for g in self.optimizer.param_groups
        ]
        if copy:
            shared_param_grad = deepcopy(shared_param_grad)
        return shared_param_grad

    def weight_grad(self, grad_sim):
        for g_param, g_sim in zip(self.optimizer.param_groups, grad_sim):
            for p_param, p_sim in zip(g_param["params"], g_sim):
                p_param.grad *= p_sim

    def grad_sim(self, grad_a, grad_b, reduce=True):
        assert self.grad_sim_metric in ["cos", "fisher_cos", "dot_product"]
        if "fisher" in self.grad_sim_metric.split("_"):
            grad_a = [[p ** 2 for p in g] for g in grad_a]
            grad_b = [[p ** 2 for p in g] for g in grad_b]

        grad_sim = [
            [torch.sum(p_a * p_b) for p_a, p_b in zip(g_a, g_b)] for g_a, g_b in zip(grad_a, grad_b)
        ]
        if reduce:
            # bk_grad_sim = grad_sim
            grad_sim = [[sum([sum(g) for g in grad_sim])]]
            # grad_pc = [[p / grad_sim[0][0] for p in g] for g in bk_grad_sim]
            # grad_pc = torch.stack(grad_pc[0] + grad_pc[2])

        if "cos" in self.grad_sim_metric.split("_"):
            sqr_a = [[(p ** 2).sum() for p in g] for g in grad_a]
            sqr_b = [[(p ** 2).sum() for p in g] for g in grad_b]
            if reduce:
                # bk_sqr_a = sqr_a
                # bk_sqr_b = sqr_b
                sqr_a = [[sum([sum(g) for g in sqr_a])]]
                sqr_b = [[sum([sum(g) for g in sqr_b])]]
                # sqr_a_pc = [[p / sqr_a[0][0] for p in g] for g in bk_sqr_a]
                # sqr_a_pc = torch.stack(sqr_a_pc[0] + sqr_a_pc[2])
                # sqr_b_pc = [[p / sqr_b[0][0] for p in g] for g in bk_sqr_b]
                # sqr_b_pc = torch.stack(sqr_b_pc[0] + sqr_b_pc[2])
            grad_sim = [
                [
                    sim / (torch.sqrt(a) * torch.sqrt(b) + 1e-10)
                    for sim, a, b in zip(g_sim, g_a, g_b)
                ]
                for g_sim, g_a, g_b in zip(grad_sim, sqr_a, sqr_b)
            ]

        grad_sim = [[self.nonlinear_fn(p) for p in g] for g in grad_sim]
        # self.count += 1
        # if self.count in [1, 100, 200, 400, 600, 800] or self.count % 1000 == 0:
        #     full_list = [
        #         "encoder.embeddings.word_embeddings.weight",
        #         "encoder.embeddings.position_embeddings.weight",
        #         "encoder.embeddings.token_type_embeddings.weight",
        #         "encoder.encoder.layer.0.attention.self.query.weight",
        #         "encoder.encoder.layer.0.attention.self.key.weight",
        #         "encoder.encoder.layer.0.attention.self.value.weight",
        #         "encoder.encoder.layer.0.attention.output.dense.weight",
        #         "encoder.encoder.layer.0.intermediate.dense.weight",
        #         "encoder.encoder.layer.0.output.dense.weight",
        #         "encoder.encoder.layer.1.attention.self.query.weight",
        #         "encoder.encoder.layer.1.attention.self.key.weight",
        #         "encoder.encoder.layer.1.attention.self.value.weight",
        #         "encoder.encoder.layer.1.attention.output.dense.weight",
        #         "encoder.encoder.layer.1.intermediate.dense.weight",
        #         "encoder.encoder.layer.1.output.dense.weight",
        #         "encoder.encoder.layer.2.attention.self.query.weight",
        #         "encoder.encoder.layer.2.attention.self.key.weight",
        #         "encoder.encoder.layer.2.attention.self.value.weight",
        #         "encoder.encoder.layer.2.attention.output.dense.weight",
        #         "encoder.encoder.layer.2.intermediate.dense.weight",
        #         "encoder.encoder.layer.2.output.dense.weight",
        #         "encoder.encoder.layer.3.attention.self.query.weight",
        #         "encoder.encoder.layer.3.attention.self.key.weight",
        #         "encoder.encoder.layer.3.attention.self.value.weight",
        #         "encoder.encoder.layer.3.attention.output.dense.weight",
        #         "encoder.encoder.layer.3.intermediate.dense.weight",
        #         "encoder.encoder.layer.3.output.dense.weight",
        #         "encoder.encoder.layer.4.attention.self.query.weight",
        #         "encoder.encoder.layer.4.attention.self.key.weight",
        #         "encoder.encoder.layer.4.attention.self.value.weight",
        #         "encoder.encoder.layer.4.attention.output.dense.weight",
        #         "encoder.encoder.layer.4.intermediate.dense.weight",
        #         "encoder.encoder.layer.4.output.dense.weight",
        #         "encoder.encoder.layer.5.attention.self.query.weight",
        #         "encoder.encoder.layer.5.attention.self.key.weight",
        #         "encoder.encoder.layer.5.attention.self.value.weight",
        #         "encoder.encoder.layer.5.attention.output.dense.weight",
        #         "encoder.encoder.layer.5.intermediate.dense.weight",
        #         "encoder.encoder.layer.5.output.dense.weight",
        #         "encoder.encoder.layer.6.attention.self.query.weight",
        #         "encoder.encoder.layer.6.attention.self.key.weight",
        #         "encoder.encoder.layer.6.attention.self.value.weight",
        #         "encoder.encoder.layer.6.attention.output.dense.weight",
        #         "encoder.encoder.layer.6.intermediate.dense.weight",
        #         "encoder.encoder.layer.6.output.dense.weight",
        #         "encoder.encoder.layer.7.attention.self.query.weight",
        #         "encoder.encoder.layer.7.attention.self.key.weight",
        #         "encoder.encoder.layer.7.attention.self.value.weight",
        #         "encoder.encoder.layer.7.attention.output.dense.weight",
        #         "encoder.encoder.layer.7.intermediate.dense.weight",
        #         "encoder.encoder.layer.7.output.dense.weight",
        #         "encoder.encoder.layer.8.attention.self.query.weight",
        #         "encoder.encoder.layer.8.attention.self.key.weight",
        #         "encoder.encoder.layer.8.attention.self.value.weight",
        #         "encoder.encoder.layer.8.attention.output.dense.weight",
        #         "encoder.encoder.layer.8.intermediate.dense.weight",
        #         "encoder.encoder.layer.8.output.dense.weight",
        #         "encoder.encoder.layer.9.attention.self.query.weight",
        #         "encoder.encoder.layer.9.attention.self.key.weight",
        #         "encoder.encoder.layer.9.attention.self.value.weight",
        #         "encoder.encoder.layer.9.attention.output.dense.weight",
        #         "encoder.encoder.layer.9.intermediate.dense.weight",
        #         "encoder.encoder.layer.9.output.dense.weight",
        #         "encoder.encoder.layer.10.attention.self.query.weight",
        #         "encoder.encoder.layer.10.attention.self.key.weight",
        #         "encoder.encoder.layer.10.attention.self.value.weight",
        #         "encoder.encoder.layer.10.attention.output.dense.weight",
        #         "encoder.encoder.layer.10.intermediate.dense.weight",
        #         "encoder.encoder.layer.10.output.dense.weight",
        #         "encoder.encoder.layer.11.attention.self.query.weight",
        #         "encoder.encoder.layer.11.attention.self.key.weight",
        #         "encoder.encoder.layer.11.attention.self.value.weight",
        #         "encoder.encoder.layer.11.attention.output.dense.weight",
        #         "encoder.encoder.layer.11.intermediate.dense.weight",
        #         "encoder.encoder.layer.11.output.dense.weight",
        #         "encoder.embeddings.LayerNorm.weight",
        #         "encoder.embeddings.LayerNorm.bias",
        #         "encoder.encoder.layer.0.attention.self.query.bias",
        #         "encoder.encoder.layer.0.attention.self.key.bias",
        #         "encoder.encoder.layer.0.attention.self.value.bias",
        #         "encoder.encoder.layer.0.attention.output.dense.bias",
        #         "encoder.encoder.layer.0.attention.output.LayerNorm.weight",
        #         "encoder.encoder.layer.0.attention.output.LayerNorm.bias",
        #         "encoder.encoder.layer.0.intermediate.dense.bias",
        #         "encoder.encoder.layer.0.output.dense.bias",
        #         "encoder.encoder.layer.0.output.LayerNorm.weight",
        #         "encoder.encoder.layer.0.output.LayerNorm.bias",
        #         "encoder.encoder.layer.1.attention.self.query.bias",
        #         "encoder.encoder.layer.1.attention.self.key.bias",
        #         "encoder.encoder.layer.1.attention.self.value.bias",
        #         "encoder.encoder.layer.1.attention.output.dense.bias",
        #         "encoder.encoder.layer.1.attention.output.LayerNorm.weight",
        #         "encoder.encoder.layer.1.attention.output.LayerNorm.bias",
        #         "encoder.encoder.layer.1.intermediate.dense.bias",
        #         "encoder.encoder.layer.1.output.dense.bias",
        #         "encoder.encoder.layer.1.output.LayerNorm.weight",
        #         "encoder.encoder.layer.1.output.LayerNorm.bias",
        #         "encoder.encoder.layer.2.attention.self.query.bias",
        #         "encoder.encoder.layer.2.attention.self.key.bias",
        #         "encoder.encoder.layer.2.attention.self.value.bias",
        #         "encoder.encoder.layer.2.attention.output.dense.bias",
        #         "encoder.encoder.layer.2.attention.output.LayerNorm.weight",
        #         "encoder.encoder.layer.2.attention.output.LayerNorm.bias",
        #         "encoder.encoder.layer.2.intermediate.dense.bias",
        #         "encoder.encoder.layer.2.output.dense.bias",
        #         "encoder.encoder.layer.2.output.LayerNorm.weight",
        #         "encoder.encoder.layer.2.output.LayerNorm.bias",
        #         "encoder.encoder.layer.3.attention.self.query.bias",
        #         "encoder.encoder.layer.3.attention.self.key.bias",
        #         "encoder.encoder.layer.3.attention.self.value.bias",
        #         "encoder.encoder.layer.3.attention.output.dense.bias",
        #         "encoder.encoder.layer.3.attention.output.LayerNorm.weight",
        #         "encoder.encoder.layer.3.attention.output.LayerNorm.bias",
        #         "encoder.encoder.layer.3.intermediate.dense.bias",
        #         "encoder.encoder.layer.3.output.dense.bias",
        #         "encoder.encoder.layer.3.output.LayerNorm.weight",
        #         "encoder.encoder.layer.3.output.LayerNorm.bias",
        #         "encoder.encoder.layer.4.attention.self.query.bias",
        #         "encoder.encoder.layer.4.attention.self.key.bias",
        #         "encoder.encoder.layer.4.attention.self.value.bias",
        #         "encoder.encoder.layer.4.attention.output.dense.bias",
        #         "encoder.encoder.layer.4.attention.output.LayerNorm.weight",
        #         "encoder.encoder.layer.4.attention.output.LayerNorm.bias",
        #         "encoder.encoder.layer.4.intermediate.dense.bias",
        #         "encoder.encoder.layer.4.output.dense.bias",
        #         "encoder.encoder.layer.4.output.LayerNorm.weight",
        #         "encoder.encoder.layer.4.output.LayerNorm.bias",
        #         "encoder.encoder.layer.5.attention.self.query.bias",
        #         "encoder.encoder.layer.5.attention.self.key.bias",
        #         "encoder.encoder.layer.5.attention.self.value.bias",
        #         "encoder.encoder.layer.5.attention.output.dense.bias",
        #         "encoder.encoder.layer.5.attention.output.LayerNorm.weight",
        #         "encoder.encoder.layer.5.attention.output.LayerNorm.bias",
        #         "encoder.encoder.layer.5.intermediate.dense.bias",
        #         "encoder.encoder.layer.5.output.dense.bias",
        #         "encoder.encoder.layer.5.output.LayerNorm.weight",
        #         "encoder.encoder.layer.5.output.LayerNorm.bias",
        #         "encoder.encoder.layer.6.attention.self.query.bias",
        #         "encoder.encoder.layer.6.attention.self.key.bias",
        #         "encoder.encoder.layer.6.attention.self.value.bias",
        #         "encoder.encoder.layer.6.attention.output.dense.bias",
        #         "encoder.encoder.layer.6.attention.output.LayerNorm.weight",
        #         "encoder.encoder.layer.6.attention.output.LayerNorm.bias",
        #         "encoder.encoder.layer.6.intermediate.dense.bias",
        #         "encoder.encoder.layer.6.output.dense.bias",
        #         "encoder.encoder.layer.6.output.LayerNorm.weight",
        #         "encoder.encoder.layer.6.output.LayerNorm.bias",
        #         "encoder.encoder.layer.7.attention.self.query.bias",
        #         "encoder.encoder.layer.7.attention.self.key.bias",
        #         "encoder.encoder.layer.7.attention.self.value.bias",
        #         "encoder.encoder.layer.7.attention.output.dense.bias",
        #         "encoder.encoder.layer.7.attention.output.LayerNorm.weight",
        #         "encoder.encoder.layer.7.attention.output.LayerNorm.bias",
        #         "encoder.encoder.layer.7.intermediate.dense.bias",
        #         "encoder.encoder.layer.7.output.dense.bias",
        #         "encoder.encoder.layer.7.output.LayerNorm.weight",
        #         "encoder.encoder.layer.7.output.LayerNorm.bias",
        #         "encoder.encoder.layer.8.attention.self.query.bias",
        #         "encoder.encoder.layer.8.attention.self.key.bias",
        #         "encoder.encoder.layer.8.attention.self.value.bias",
        #         "encoder.encoder.layer.8.attention.output.dense.bias",
        #         "encoder.encoder.layer.8.attention.output.LayerNorm.weight",
        #         "encoder.encoder.layer.8.attention.output.LayerNorm.bias",
        #         "encoder.encoder.layer.8.intermediate.dense.bias",
        #         "encoder.encoder.layer.8.output.dense.bias",
        #         "encoder.encoder.layer.8.output.LayerNorm.weight",
        #         "encoder.encoder.layer.8.output.LayerNorm.bias",
        #         "encoder.encoder.layer.9.attention.self.query.bias",
        #         "encoder.encoder.layer.9.attention.self.key.bias",
        #         "encoder.encoder.layer.9.attention.self.value.bias",
        #         "encoder.encoder.layer.9.attention.output.dense.bias",
        #         "encoder.encoder.layer.9.attention.output.LayerNorm.weight",
        #         "encoder.encoder.layer.9.attention.output.LayerNorm.bias",
        #         "encoder.encoder.layer.9.intermediate.dense.bias",
        #         "encoder.encoder.layer.9.output.dense.bias",
        #         "encoder.encoder.layer.9.output.LayerNorm.weight",
        #         "encoder.encoder.layer.9.output.LayerNorm.bias",
        #         "encoder.encoder.layer.10.attention.self.query.bias",
        #         "encoder.encoder.layer.10.attention.self.key.bias",
        #         "encoder.encoder.layer.10.attention.self.value.bias",
        #         "encoder.encoder.layer.10.attention.output.dense.bias",
        #         "encoder.encoder.layer.10.attention.output.LayerNorm.weight",
        #         "encoder.encoder.layer.10.attention.output.LayerNorm.bias",
        #         "encoder.encoder.layer.10.intermediate.dense.bias",
        #         "encoder.encoder.layer.10.output.dense.bias",
        #         "encoder.encoder.layer.10.output.LayerNorm.weight",
        #         "encoder.encoder.layer.10.output.LayerNorm.bias",
        #         "encoder.encoder.layer.11.attention.self.query.bias",
        #         "encoder.encoder.layer.11.attention.self.key.bias",
        #         "encoder.encoder.layer.11.attention.self.value.bias",
        #         "encoder.encoder.layer.11.attention.output.dense.bias",
        #         "encoder.encoder.layer.11.attention.output.LayerNorm.weight",
        #         "encoder.encoder.layer.11.attention.output.LayerNorm.bias",
        #         "encoder.encoder.layer.11.intermediate.dense.bias",
        #         "encoder.encoder.layer.11.output.dense.bias",
        #         "encoder.encoder.layer.11.output.LayerNorm.weight",
        #         "encoder.encoder.layer.11.output.LayerNorm.bias",
        #     ]
        #     print(f"\niter: {self.count}")
        #     print(f"grad_sim: {grad_sim[0][0].item()}")
        #     top5 = torch.topk(grad_pc, 5, dim=0)[1]
        #     for ind in top5:
        #         print(
        #             f"param:{full_list[ind]}, grad_sim: {grad_pc[ind]}, sqr_a: {sqr_a_pc[ind]}, sqr_b: {sqr_b_pc[ind]}"
        #         )
        #     grad_a_temp = grad_a[0] + grad_a[2]
        #     grad_b_temp = grad_b[0] + grad_b[2]
        #     t = grad_a_temp[top5[0]] * grad_b_temp[top5[0]]
        #     t = t / t.sum()

        #     if len(t.squeeze().size()) == 2:
        #         width = t.size()[1]
        #         values, indices = torch.topk(t.flatten(), 100)
        #         indices = [(i // width, i % width) for i in indices.tolist()]
        #         print(f"largest elements in matrix \n percentage:{values} \n indices {indices}")
        #     else:
        #         values, indices = torch.topk(t, 20)
        #         indices = indices.tolist()
        #         print(f"largest elements in vector \n percentage:{values} \n indices {indices}")
        #     print("", flush=True)

        if reduce:
            return grad_sim[0][0]
        else:
            return grad_sim


class OptimizerSchedulerForReptile(OptimizerScheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def create_optimizer(
    args,
    model,
    learning_rate,
    t_total,
    warmup_steps,
    warmup_proportion,
    optimizer_epsilon=1e-8,
    optimizer_type="adam",
    verbose=False,
):
    return create_optimizer_from_params(
        args=args,
        named_parameters=list(model.named_parameters()),
        learning_rate=learning_rate,
        t_total=t_total,
        warmup_steps=warmup_steps,
        warmup_proportion=warmup_proportion,
        optimizer_epsilon=optimizer_epsilon,
        optimizer_type=optimizer_type,
        verbose=verbose,
    )


def create_optimizer_from_params(
    args,
    named_parameters,
    learning_rate,
    t_total,
    warmup_steps,
    warmup_proportion,
    optimizer_epsilon=1e-8,
    optimizer_type="adam",
    scheduler_type="linear",
    verbose=False,
):
    # Prepare optimizer
    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "adapter.down_project.weight",
        "adapter.up_project.weight",
        "weighted_sum.weights",
        "task_sharing",
        "layer_sharing",
    ]
    if verbose:
        print("No optimizer decay for:")
        for n, p in named_parameters:
            if any(nd in n for nd in no_decay):
                print(f"  {n}")

    used_named_parameters = [(n, p) for n, p in named_parameters if p.requires_grad]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in used_named_parameters
                if (n.startswith("encoder.e") or n.startswith("dds_model.encoder.e"))
                and (not any(nd in n for nd in no_decay))
            ],
            "names": [
                n
                for n, p in used_named_parameters
                if (n.startswith("encoder.e") or n.startswith("dds_model.encoder.e"))
                and (not any(nd in n for nd in no_decay))
            ],
            "weight_decay": 0.01,
            "shared": True,
            "is_base_encoder": [
                n.startswith("encoder.e")
                for n, p in used_named_parameters
                if (n.startswith("encoder.e") or n.startswith("dds_model.encoder.e"))
                and (not any(nd in n for nd in no_decay))
            ],
        },
        {
            "params": [
                p
                for n, p in used_named_parameters
                if (not n.startswith("encoder.e") and not n.startswith("dds_model.encoder.e"))
                and (not any(nd in n for nd in no_decay))
            ],
            "names": [
                n
                for n, p in used_named_parameters
                if (not n.startswith("encoder.e") and not n.startswith("dds_model.encoder.e"))
                and (not any(nd in n for nd in no_decay))
            ],
            "weight_decay": 0.005,
            "shared": False,
            "is_base_encoder": None,
        },
        {
            "params": [
                p
                for n, p in used_named_parameters
                if (n.startswith("encoder.e") or n.startswith("dds_model.encoder.e"))
                and any(nd in n for nd in no_decay)
            ],
            "names": [
                n
                for n, p in used_named_parameters
                if (n.startswith("encoder.e") or n.startswith("dds_model.encoder.e"))
                and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "shared": True,
            "is_base_encoder": [
                n.startswith("encoder.e")
                for n, p in used_named_parameters
                if (n.startswith("encoder.e") or n.startswith("dds_model.encoder.e"))
                and any(nd in n for nd in no_decay)
            ],
        },
        {
            "params": [
                p
                for n, p in used_named_parameters
                if (not n.startswith("encoder.e") and not n.startswith("dds_model.encoder.e"))
                and any(nd in n for nd in no_decay)
            ],
            "names": [
                n
                for n, p in used_named_parameters
                if (not n.startswith("encoder.e") and not n.startswith("dds_model.encoder.e"))
                and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "shared": False,
            "is_base_encoder": None,
        },
    ]

    if optimizer_type == "adam":
        if verbose:
            print("Using AdamW")
        optimizer = transformers.AdamW(
            optimizer_grouped_parameters, lr=learning_rate, eps=optimizer_epsilon
        )
    elif optimizer_type == "radam":
        if verbose:
            print("Using RAdam")
        optimizer = RAdam(optimizer_grouped_parameters, lr=learning_rate, eps=optimizer_epsilon)
    else:
        raise KeyError(optimizer_type)

    if scheduler_type == "linear":
        warmup_steps = resolve_warmup_steps(
            t_total=t_total, warmup_steps=warmup_steps, warmup_proportion=warmup_proportion,
        )
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )
    else:
        raise KeyError(scheduler_type)

    if args.runner_type in ["default", "distill"]:
        optimizer_scheduler = OptimizerScheduler(optimizer=optimizer, scheduler=scheduler)
    elif args.runner_type in ["multidds", "dds", "grad_sim"]:
        optimizer_scheduler = OptimizerSchedulerWithGradOps(
            optimizer=optimizer,
            scheduler=scheduler,
            grad_sim_metric=args.grad_sim_metric,
            grad_sim_nonlinear=args.grad_sim_nonlinear,
        )
    elif args.runner_type == "reptile":
        optimizer_scheduler = OptimizerSchedulerForReptile(optimizer=optimizer, scheduler=scheduler)
    else:
        raise KeyError(args.runner_type)
    return optimizer_scheduler


def resolve_warmup_steps(t_total, warmup_steps, warmup_proportion):
    if warmup_steps is None and warmup_proportion is None:
        raise RuntimeError()
    elif warmup_steps is not None and warmup_proportion is not None:
        raise RuntimeError()
    elif warmup_steps is None and warmup_proportion is not None:
        return warmup_proportion * t_total
    elif warmup_steps is not None and warmup_proportion is None:
        return warmup_steps
    else:
        raise RuntimeError()


def fp16ize(model, optimizer, fp16_opt_level):
    try:
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        from apex import amp
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
        )
    model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
    return model, optimizer


def parallelize_gpu(model):
    return torch.nn.DataParallel(model)


def parallelize_dist(model, local_rank):
    return torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank,
    )


def raw_special_model_setup(model, optimizer, fp16, fp16_opt_level, n_gpu, local_rank):
    """Perform setup for special modes (e.g., FP16, DataParallel, and/or DistributedDataParallel.

    Args:
        model (nn.Module): torch model object.
        optimizer: TODO
        fp16 (bool): True to enable FP16 mode.
        fp16_opt_level (str): Apex AMP optimization level default mode identifier.
        n_gpu: number of GPUs.
        local_rank (int): Which GPU the script should use in DistributedDataParallel mode.

    Notes:
        Initialization steps performed in init_cuda_from_args() set n_gpu = 1 when local_rank != -1.

    Returns:
        Model and optimizer with the specified special configuration.

    """
    if fp16:
        model, optimizer = fp16ize(model=model, optimizer=optimizer, fp16_opt_level=fp16_opt_level)
    if n_gpu > 1:
        model = parallelize_gpu(model=model)
    if local_rank != -1:
        model = parallelize_dist(model=model, local_rank=local_rank)
    return model, optimizer


def special_model_setup(
    model_wrapper, optimizer_scheduler, fp16, fp16_opt_level, n_gpu, local_rank
):
    model, optimizer = raw_special_model_setup(
        model=model_wrapper.model,
        optimizer=optimizer_scheduler.optimizer,
        fp16=fp16,
        fp16_opt_level=fp16_opt_level,
        n_gpu=n_gpu,
        local_rank=local_rank,
    )
    model_wrapper.model = model
    optimizer_scheduler.optimizer = optimizer
