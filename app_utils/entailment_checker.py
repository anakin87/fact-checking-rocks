from typing import List, Optional

from transformers import AutoModelForSequenceClassification,AutoTokenizer,AutoConfig
import torch
from haystack.nodes.base import BaseComponent
from haystack.modeling.utils import initialize_device_settings
from haystack.schema import Document, Answer, Span

class EntailmentChecker(BaseComponent):
    """
    This node checks the entailment between every document content and the query.
    It enrichs the documents metadata with entailment_info
    """

    outgoing_edges = 1

    def __init__(
        self,
        model_name_or_path: str = "roberta-large-mnli",
        model_version: Optional[str] = None,
        tokenizer: Optional[str] = None,
        use_gpu: bool = True,
        batch_size: int = 16,
    ):
        """
        Load a Natural Language Inference model from Transformers.

        :param model_name_or_path: Directory of a saved model or the name of a public model.
        See https://huggingface.co/models for full list of available models.
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param tokenizer: Name of the tokenizer (usually the same as model)
        :param use_gpu: Whether to use GPU (if available).
        # :param batch_size: Number of Documents to be processed at a time.
        """
        super().__init__()

        self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=False)

        tokenizer = tokenizer or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path,revision=model_version)
        self.batch_size = batch_size
        self.model.to(str(self.devices[0]))
        
        id2label = AutoConfig.from_pretrained(model_name_or_path).id2label
        self.labels= [id2label[k].lower() for k in sorted(id2label)]
        if 'entailment' not in self.labels:
            raise ValueError("The model config must contain entailment value in the id2label dict.")
    
    def run(self, query: str, documents: List[Document]):
        for doc in documents:
            entailment_dict=self.get_entailment(premise=doc.content, hypotesis=query)
            doc.meta['entailment_info']=entailment_dict
        return {'documents':documents}, "output_1"
    
    def run_batch():
        pass
    
    def get_entailment(self, premise,hypotesis):
        with torch.no_grad():
            inputs = self.tokenizer(f'{premise}{self.tokenizer.sep_token}{hypotesis}', return_tensors="pt").to(self.devices[0])
            out = self.model(**inputs)
            logits = out.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0,:].cpu().detach().numpy()
        entailment_dict={k.lower():v for k,v in zip (self.labels, probs)}
        return entailment_dict