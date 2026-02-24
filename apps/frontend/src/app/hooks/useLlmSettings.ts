// src/app/hooks/useLlmSettings.ts
import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchLlmModelSettings, updateLlmModelSettings } from "../api/settingsApi";
import { LLM_MODEL_IDS, normalizeLlmModel, requiresApiKey } from "../llm/llmMeta";
import { useLocalStorageState } from "./useLocalStorageState";

type ChangeModelResult =
  | { ok: true; model: string }
  | { ok: false; reason: "api_key_required" | "invalid_model" | "unknown" };

const MODEL_STORAGE_OPTIONS = {
  serialize: (v: string) => v,
  deserialize: (raw: string) => normalizeLlmModel(raw),
};

const APIKEY_STORAGE_OPTIONS = {
  serialize: (v: string | null) => v ?? "",
  deserialize: (raw: string) => (raw ? raw : null),
};

export function useLlmSettings() {
  // 전역 상태
  const [activeModel, setActiveModel] = useLocalStorageState<string>(
    "activeModel",
    "gemma3-12b",
    MODEL_STORAGE_OPTIONS,
  );

  const [apiKey, setApiKey] = useLocalStorageState<string | null>(
    "geminiApiKey",
    null,
    APIKEY_STORAGE_OPTIONS,
  );

  // UI 상태
  const [llmModels, setLlmModels] = useState<string[]>([...LLM_MODEL_IDS]);
  const [modelSyncing, setModelSyncing] = useState(false);

  const isApiKeyRequired = useMemo(() => requiresApiKey(activeModel), [activeModel]);
  const hasRequiredApiKey = useMemo(() => (isApiKeyRequired ? !!apiKey : true), [isApiKeyRequired, apiKey]);

  // 백엔드 settings 동기화: 실패해도 프론트는 3개 모델로 동작
  useEffect(() => {
    const ac = new AbortController();
    void (async () => {
      try {
        const cfg = await fetchLlmModelSettings({ signal: ac.signal });
        const models = Array.isArray(cfg.available_models) ? cfg.available_models : [];
        const filtered = models.filter((m) => LLM_MODEL_IDS.includes(m as any));
        setLlmModels(filtered.length ? filtered : [...LLM_MODEL_IDS]);

        if (cfg.active_model) {
          setActiveModel(normalizeLlmModel(cfg.active_model));
        }
      } catch {
        setLlmModels([...LLM_MODEL_IDS]);
      }
    })();

    return () => ac.abort();
  }, [setActiveModel]);

  const changeModel = useCallback(
    async (nextModel: string): Promise<ChangeModelResult> => {
      if (!nextModel) return { ok: false, reason: "unknown" };
      if (!LLM_MODEL_IDS.includes(nextModel as any)) return { ok: false, reason: "invalid_model" };

      if (requiresApiKey(nextModel) && !apiKey) {
        return { ok: false, reason: "api_key_required" };
      }

      // 프론트 전역 상태 즉시 반영(전역 적용)
      setActiveModel(normalizeLlmModel(nextModel));
      setModelSyncing(true);

      try {
        // 백엔드가 아직 준비 안 돼도 catch에서 프론트 상태는 유지됩니다.
        const updated = await updateLlmModelSettings(nextModel);
        const normalized = normalizeLlmModel(updated.active_model || nextModel);
        setActiveModel(normalized);

        if (Array.isArray(updated.available_models) && updated.available_models.length > 0) {
          const filtered = updated.available_models.filter((m: string) => LLM_MODEL_IDS.includes(m as any));
          setLlmModels(filtered.length ? filtered : [...LLM_MODEL_IDS]);
        }

        return { ok: true, model: normalized };
      } catch {
        // 백엔드 실패해도 프론트는 전역 상태 유지
        return { ok: true, model: normalizeLlmModel(nextModel) };
      } finally {
        setModelSyncing(false);
      }
    },
    [apiKey, setActiveModel],
  );

  return {
    activeModel,
    llmModels,
    modelSyncing,

    apiKey,
    setApiKey,

    isApiKeyRequired,
    hasRequiredApiKey,

    changeModel,
  };
}