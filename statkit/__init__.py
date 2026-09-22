"""StatKit — a pure-Python statistics package.

No AI/LLM, no network, and no disk I/O ever run from inside this package;
that guarantee is enforced mechanically by tests/test_no_ai_no_io.py. This
package also imports zero Streamlit (tests/test_zero_streamlit.py); the UI
lives entirely in the top-level app.py.
"""
