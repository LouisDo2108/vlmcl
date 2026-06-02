"""Quick smoke test (run from vlmcl/src/tevatron): python -m tevatron.hyperbolic.test"""

from tevatron.hyperbolic.collator import CLIPCollator
from tevatron.hyperbolic.dataset import CLIPTrainDataset
from tevatron.hyperbolic.model import CLIPContrastiveModel
from tevatron.hyperbolic.utils import print_master
from transformers import CLIPProcessor


def load_clip_processor(model_args):
    model_name = model_args.processor_name or model_args.model_name_or_path
    print_master(f"Loading CLIPProcessor from {model_name}")
    return CLIPProcessor.from_pretrained(model_name)


def main():
    from tevatron.hyperbolic.arguments import DataArguments, ModelArguments

    model_args = ModelArguments(model_name_or_path="openai/clip-vit-large-patch14", lora=False)
    processor = load_clip_processor(model_args)

    # model = CLIPContrastiveModel.build(model_args)

    # batch = [
    #     (("a photo of a cat", None), ("a photo of a dog", None)),
    # ]
    # data_args = DataArguments(max_len=77)
    # qry, tgt = CLIPCollator(data_args, processor)(batch)
    # with __import__("torch").no_grad():
    #     out = model(qry=qry, tgt=tgt)
    # print("loss", float(out))


if __name__ == "__main__":
    main()
