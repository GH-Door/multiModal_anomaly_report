// src/app/utils/labels.ts

export function decisionLabel(decision?: string) {
  const d = (decision ?? "").trim().toLowerCase();
  if (d === "ok" || d === "normal") return "정상 (OK)";
  if (d === "ng" || d === "anomaly") return "불량 (NG)";
  if (d === "review") return "재검토 (REVIEW)";
  return decision || "미분류";
}

export function defectTypeLabel(type?: string) {
  const t = (type ?? "").trim().toLowerCase();
  const map: Record<string, string> = {
    seal_issue: "실링 불량",
    contamination: "오염",
    crack: "파손/균열",
    missing_component: "구성요소 누락",
    scratch: "스크래치",
  };
  return map[t] ?? (type || "-");
}

export function severityLabel(sev?: string) {
  const s = (sev ?? "").trim().toLowerCase();
  if (s === "high") return "높음";
  if (s === "med" || s === "medium") return "중간";
  if (s === "low") return "낮음";
  return sev || "-";
}

export function locationLabel(loc?: string) {
  const l = (loc ?? "").trim().toLowerCase();
  const map: Record<string, string> = {
    "top-right": "상단 우측",
    "top-left": "상단 좌측",
    "bottom-right": "하단 우측",
    "bottom-left": "하단 좌측",
    center: "중앙",
  };
  return map[l] ?? (loc || "-");
}
