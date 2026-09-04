"""API/llm import must not cycle through article_ingestion package init."""


def test_api_app_imports_without_llm_circular():
    from knowledge_engine.api.app import create_app
    from knowledge_engine.llm import invoke_logged, structured_chat

    assert callable(invoke_logged)
    assert callable(structured_chat)
    app = create_app()
    assert app.title
