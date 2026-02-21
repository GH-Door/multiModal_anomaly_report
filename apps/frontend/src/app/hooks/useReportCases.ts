// src/app/hooks/useReportCases.ts
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchReportsAll } from "../api/reportsApi";
import { mapReportsToAnomalyCases } from "../data/reportMapper";
import type { AnomalyCase } from "../data/mockData";
import type { ReportListQuery } from "../api/reportsApi";

type UseReportCasesState = {
  loading: boolean;
  error: unknown;
  cases: AnomalyCase[];
};

export function useReportCases(params?: {
  query?: Omit<ReportListQuery, "limit" | "offset">;
  pageSize?: number;
  maxItems?: number;
}) {
  const [state, setState] = useState<UseReportCasesState>({
    loading: true,
    error: null,
    cases: [],
  });

  const lastAbortRef = useRef<AbortController | null>(null);

  const load = useCallback(async () => {
    lastAbortRef.current?.abort();
    const ac = new AbortController();
    lastAbortRef.current = ac;

    setState((s) => ({ ...s, loading: true, error: null }));

    try {
      const reports = await fetchReportsAll(params?.query, {
        signal: ac.signal,
        pageSize: params?.pageSize,
        maxItems: params?.maxItems,
      });
      const mapped = mapReportsToAnomalyCases(reports);
      setState({ loading: false, error: null, cases: mapped });
    } catch (err) {
      if ((err as any)?.name === "AbortError") return;
      setState({ loading: false, error: err, cases: [] });
    }
  }, [params?.query, params?.pageSize, params?.maxItems]);

  useEffect(() => {
    load();
    return () => lastAbortRef.current?.abort();
  }, [load]);

  return useMemo(
    () => ({
      ...state,
      refetch: load,
    }),
    [state, load]
  );
}
