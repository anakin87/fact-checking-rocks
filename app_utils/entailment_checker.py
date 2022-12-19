from typing import List, Optional

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch
from haystack.nodes.base import BaseComponent
from haystack.modeling.utils import initialize_device_settings
from haystack.schema import Document


class EntailmentChecker(BaseComponent):
    """
    This node checks the entailment between every document content and the query.
    It enrichs the documents metadata with entailment informations.
    It also returns aggregate entailment information.
    """

    outgoing_edges = 1

    def __init__(
        self,
        model_name_or_path: str = "roberta-large-mnli",
        model_version: Optional[str] = None,
        tokenizer: Optional[str] = None,
        use_gpu: bool = True,
        batch_size: int = 16,
        entailment_contradiction_threshold: float = 0.5,
    ):
        """
        Load a Natural Language Inference model from Transformers.

        :param model_name_or_path: Directory of a saved model or the name of a public model.
        See https://huggingface.co/models for full list of available models.
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param tokenizer: Name of the tokenizer (usually the same as model)
        :param use_gpu: Whether to use GPU (if available).
        :param batch_size: Number of Documents to be processed at a time.
        :param entailment_contradiction_threshold: if in the first N documents there is a strong evidence of entailment/contradiction
        (aggregate entailment or contradiction are greater than the threshold), the less relevant documents are not taken into account
        """
        super().__init__()

        self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=False)

        tokenizer = tokenizer or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path, revision=model_version
        )
        self.batch_size = batch_size
        self.entailment_contradiction_threshold = entailment_contradiction_threshold
        self.model.to(str(self.devices[0]))

        id2label = AutoConfig.from_pretrained(model_name_or_path).id2label
        self.labels = [id2label[k].lower() for k in sorted(id2label)]
        if "entailment" not in self.labels:
            raise ValueError(
                "The model config must contain entailment value in the id2label dict."
            )

    def run(self, query: str, documents: List[Document]):

        scores, agg_con, agg_neu, agg_ent = 0, 0, 0, 0
        for i, doc in enumerate(documents):
            entailment_info = self.get_entailment(premise=doc.content, hypotesis=query)
            doc.meta["entailment_info"] = entailment_info

            scores += doc.score
            con, neu, ent = (
                entailment_info["contradiction"],
                entailment_info["neutral"],
                entailment_info["entailment"],
            )
            agg_con += con * doc.score
            agg_neu += neu * doc.score
            agg_ent += ent * doc.score

            # if in the first documents there is a strong evidence of entailment/contradiction,
            # there is no need to consider less relevant documents
            if max(agg_con, agg_ent) / scores > self.entailment_contradiction_threshold:
                break

        aggregate_entailment_info = {
            "contradiction": round(agg_con / scores, 2),
            "neutral": round(agg_neu / scores, 2),
            "entailment": round(agg_ent / scores, 2),
        }

        entailment_checker_result = {
            "documents": documents[: i + 1],
            "aggregate_entailment_info": aggregate_entailment_info,
        }

        return entailment_checker_result, "output_1"

    def run_batch(self, queries: List[str], documents: List[Document]):
        pass

    def get_entailment(self, premise, hypotesis):
        with torch.no_grad():
            inputs = self.tokenizer(
                f"{premise}{self.tokenizer.sep_token}{hypotesis}", return_tensors="pt"
            ).to(self.devices[0])
            out = self.model(**inputs)
            logits = out.logits
            probs = (
                torch.nn.functional.softmax(logits, dim=-1)[0, :].cpu().detach().numpy()
            )
        entailment_dict = {k.lower(): v for k, v in zip(self.labels, probs)}
        return entailment_dict
