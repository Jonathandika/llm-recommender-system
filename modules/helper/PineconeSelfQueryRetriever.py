from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.schema.vectorstore import VectorStore
from langchain.chains.query_constructor.ir import Visitor
from langchain.schema.language_model import BaseLanguageModel
from langchain.retrievers.self_query.pinecone import PineconeTranslator
from langchain.chains.query_constructor.base import load_query_constructor_runnable

from helper.PineconeModified import PineconeModified

from typing import Any, Dict, Optional, Sequence, Type, Union

def _get_builtin_translator(vectorstore: VectorStore) -> Visitor:
    """Get the translator class corresponding to the vector store class."""
    BUILTIN_TRANSLATORS: Dict[Type[VectorStore], Type[Visitor]] = {
        PineconeModified: PineconeTranslator,
    }
    
    if vectorstore.__class__ in BUILTIN_TRANSLATORS:
        return BUILTIN_TRANSLATORS[vectorstore.__class__]()
    else:
        raise ValueError(
            f"Self query retriever with Vector Store type {vectorstore.__class__}"
            f" not supported."
        )

class PineconeSelfQueryRetriever(SelfQueryRetriever):

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        vectorstore: VectorStore,
        document_contents: str,
        metadata_field_info: Sequence[Union[AttributeInfo, dict]],
        structured_query_translator: Optional[Visitor] = None,
        chain_kwargs: Optional[Dict] = None,
        enable_limit: bool = False,
        use_original_query: bool = False,
        **kwargs: Any,
    ) -> "SelfQueryRetriever":
        if structured_query_translator is None:
            structured_query_translator = _get_builtin_translator(vectorstore)
        chain_kwargs = chain_kwargs or {}

        if "allowed_comparators" not in chain_kwargs:
            chain_kwargs[
                "allowed_comparators"
            ] = structured_query_translator.allowed_comparators
        if "allowed_operators" not in chain_kwargs:
            chain_kwargs[
                "allowed_operators"
            ] = structured_query_translator.allowed_operators
        query_constructor = load_query_constructor_runnable(
            llm,
            document_contents,
            metadata_field_info,
            enable_limit=enable_limit,
            **chain_kwargs,
        )
        return cls(
            query_constructor=query_constructor,
            vectorstore=vectorstore,
            use_original_query=use_original_query,
            structured_query_translator=structured_query_translator,
            **kwargs,
        )