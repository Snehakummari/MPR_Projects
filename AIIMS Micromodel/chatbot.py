# chatbot.py
import torch
from model_loader import (
    load_router,
    load_embedder,
    load_domain_model,
    load_domain_index,
    DEVICE
)
from retriever import retrieve_top_k, build_prompt


class HospitalChatbot:
    """
    Core chatbot class for the AIIMS Micromodel system.

    On init: loads the router (LogisticRegression), label encoder,
             and sentence embedder — all lightweight, fast.

    On ask(): routes the query → lazy-loads that department's BioBART
              model + index on first use → retrieves top-3 contexts →
              builds prompt → generates answer via BioBART.

    Department models are cached after first load so the same dept
    is fast on subsequent queries within the same session.
    """

    def __init__(self):
        print("\n══════════════════════════════════════════════════════")
        print("  AIIMS Hospital Chatbot — Initialising")
        print("══════════════════════════════════════════════════════")

        self.router, self.le = load_router()
        self.embedder        = load_embedder()

        # Cache: { domain_str -> (model, tokenizer, embeddings, qa_data) }
        self._cache = {}

        print("\n✅ Chatbot ready — department models load on first query")
        print("══════════════════════════════════════════════════════\n")

    # ── Routing ────────────────────────────────────────────────────────────────
    def _route(self, query: str) -> str:
        """Embed query → predict → decode to department label string."""
        emb        = self.embedder.encode([query], convert_to_numpy=True)
        pred_int   = self.router.predict(emb)
        department = self.le.inverse_transform(pred_int)[0]
        return department

    # ── Lazy load + cache domain assets ───────────────────────────────────────
    def _get_domain_assets(self, domain: str):
        """Load and cache model, tokenizer, embeddings, qa_data for a domain."""
        if domain not in self._cache:
            print(f"\n[Chatbot] First query for '{domain}' — loading model...")
            model, tokenizer       = load_domain_model(domain)
            embeddings, qa_data    = load_domain_index(domain)
            self._cache[domain]    = (model, tokenizer, embeddings, qa_data)
            print(f"[Chatbot] '{domain}' model cached for this session.\n")
        return self._cache[domain]

    # ── Generation ─────────────────────────────────────────────────────────────
    def _generate(self, prompt: str, model, tokenizer) -> str:
        """Tokenise prompt and run BioBART generation."""
        inputs = tokenizer(
            prompt,
            max_length      = 1024,
            truncation      = True,
            return_tensors  = 'pt'
        ).to(DEVICE)

        with torch.no_grad():
            output = model.generate(
                input_ids            = inputs['input_ids'],
                attention_mask       = inputs['attention_mask'],
                max_new_tokens       = 256,
                num_beams            = 4,
                early_stopping       = True,
                no_repeat_ngram_size = 3,
                length_penalty       = 1.0
            )

        return tokenizer.decode(output[0], skip_special_tokens=True)

    # ── Public API ─────────────────────────────────────────────────────────────
    def ask(self, query: str) -> dict:
        """
        Full pipeline: route → retrieve → generate.

        Args:
            query : raw user question string

        Returns:
            dict with keys:
                department : predicted department label
                answer     : generated answer string
                context    : list of top-3 retrieved Q&A dicts (for debug/display)
        """
        # 1. Route
        department = self._route(query)

        # 2. Load assets (cached after first use)
        model, tokenizer, embeddings, qa_data = self._get_domain_assets(department)

        # 3. Retrieve top-3 similar Q&As
        context_hits = retrieve_top_k(query, self.embedder, embeddings, qa_data, k=3)

        # 4. Build prompt
        prompt = build_prompt(query, context_hits)

        # 5. Generate answer
        answer = self._generate(prompt, model, tokenizer)

        return {
            'department' : department,
            'answer'     : answer,
            'context'    : context_hits
        }