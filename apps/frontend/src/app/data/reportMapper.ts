import type { AnomalyCase, ActionLog } from "./mockData";
import type { ReportDTO } from "../api/reportsApi";

/**
 * 백엔드 실데이터(DB) -> 프론트 AnomalyCase 매핑
 */

const PACKAGING_CLASS_LABEL: Record<string, string> = {
  cigarette_box: "cigarette box",
  drink_bottle: "drink bottle",
  drink_can: "drink can",
  food_bottle: "food_bottle",
  food_box: "food box",
  food_package: "food package",
  breakfast_box: "breakfast box",
  juice_bottle: "juice bottle",
  pushpins: "pushpins",
  screw_bag: "screw bag",
};

const LINES = ["LINE-A-01", "LINE-B-02", "LINE-C-03"];
const LOCS = ["top-left", "top-right", "bottom-left", "bottom-right", "center"] as const;

function hash01(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) / 4294967295;
}

function parseDatetime(dt?: string): Date {
  if (!dt) return new Date();
  const cleaned = dt.replace(/(\.\d{3})\d+/, "$1");
  const d = new Date(cleaned);
  return isNaN(d.getTime()) ? new Date() : d;
}

function parseJsonLike(input: unknown): any {
  if (!input) return {};
  if (typeof input === "object") return input;
  if (typeof input === "string") {
    try {
      return JSON.parse(input);
    } catch {
      return {};
    }
  }
  return {};
}

function toProductGroup(category: string): string {
  // 'GoodsAD/cigarette_box' 같은 경우 처리
  const pureCat = category.includes('/') ? category.split('/').pop()! : category;
  return PACKAGING_CLASS_LABEL[pureCat] ?? pureCat.replace(/_/g, " ");
}

function toLineId(seed: string): string {
  const v = hash01(seed);
  return LINES[Math.floor(v * LINES.length) % LINES.length];
}

function toShift(d: Date): string {
  const hour = d.getHours();
  return hour >= 7 && hour < 19 ? "주간" : "야간";
}

/**
 * 백엔드의 'ad_decision' (normal, anomaly, review_needed)을
 * 프론트엔드 UI용 (OK, NG, REVIEW)으로 변환
 */
function normalizeDecision(decisionRaw?: string): "OK" | "NG" | "REVIEW" {
  const d = (decisionRaw ?? "").trim().toLowerCase();
  if (d === "normal" || d === "ok") return "OK";
  if (d === "anomaly" || d === "ng") return "NG";
  return "REVIEW"; // review_needed 포함
}

function normalizeSeverity(raw?: string, decision?: "OK" | "NG" | "REVIEW"): "low" | "med" | "high" {
  if (decision && decision !== "NG") return "low";
  const s = (raw ?? "").trim().toLowerCase();
  if (s === "high") return "high";
  if (s === "med" || s === "medium") return "med";
  return "low";
}

function toKoreanSummary(decision: "OK" | "NG" | "REVIEW", defectType: string, location: string): string {
  if (decision === "OK") return "정상 제품으로 판정되었습니다. 이상 징후가 발견되지 않았습니다.";
  if (decision === "REVIEW") return "경계 케이스입니다. 육안 재검토를 통해 판정을 확정해 주세요.";

  const locKo =
    location === "top-left" ? "상단 좌측" :
    location === "top-right" ? "상단 우측" :
    location === "bottom-left" ? "하단 좌측" :
    location === "bottom-right" ? "하단 우측" :
    location === "center" ? "중앙" : "미상";

  return `${locKo} 영역에서 결함이 감지되었습니다. 불량으로 분류됩니다.`;
}

function toActionLog(decision: "OK" | "NG" | "REVIEW", ts: Date): ActionLog[] {
  const base = ts.getTime();
  if (decision === "OK") return [{ who: "System", when: new Date(base + 1000), what: "자동 승인" }];
  if (decision === "REVIEW") return [{ who: "박철수", when: new Date(base + 60_000), what: "재검 요청" }];
  return [
    { who: "System", when: new Date(base + 1000), what: "AI 불량 감지" },
    { who: "Operator", when: new Date(base + 60_000), what: "불량 확정" }
  ];
}

export function mapReportsToAnomalyCases(raw: any[]): AnomalyCase[] {
  return raw.map((r, idx) => {
    // 백엔드 원본 레코드/정규화 DTO 둘 다 허용
    const ts = parseDatetime(r.created_at ?? r.datetime);
    const decision = normalizeDecision(r.ad_decision ?? r.decision);

    const dataset = r.dataset ?? "UNKNOWN";
    const category = r.category ?? "unknown";

    const loc = decision === "OK" ? "none" : (r.region ?? r.location ?? "center");

    // llm_report, llm_summary는 문자열(JSON) 또는 객체(JSONB) 둘 다 가능
    const llmReport: any = parseJsonLike(r.llm_report);
    const llmSummaryObj: any = parseJsonLike(r.llm_summary);

    const defectType = llmReport.anomaly_type ?? (r.has_defect ? "anomaly" : "none");
    const severity = normalizeSeverity(llmReport.severity, decision);

    return {
      id: `CASE-${r.id || idx}`,
      timestamp: ts,
      line_id: r.line ?? toLineId(`${dataset}-${category}`),
      shift: toShift(ts),

      product_group: toProductGroup(category),
      image_id: r.image_path ? r.image_path.split('/').pop() : `img_${idx}.jpg`,

      // 이미지 경로 매핑
      image_path: r.image_path ?? undefined,
      heatmap_path: r.heatmap_path ?? undefined,
      overlay_path: r.overlay_path ?? undefined,

      decision,
      anomaly_score: typeof r.ad_score === "number" ? r.ad_score : (typeof r.confidence === "number" ? r.confidence : 0),
      threshold: (r.applied_policy?.t_high) ?? 0.7,

      defect_type: defectType,
      defect_confidence: r.area_ratio ?? (typeof r.confidence === "number" ? r.confidence : 0),
      location: loc,
      affected_area_pct: r.area_ratio ? r.area_ratio * 100 : 0,
      severity,

      model_name: "EfficientAD",
      model_version: "v1.0.0",
      inference_time_ms:
        typeof r.ad_inference_duration === "number"
          ? Math.round(r.ad_inference_duration * 1000)
          : (typeof r.inference_time === "number" ? Math.round(r.inference_time * 1000) : 0),

      // LLM 요약 매핑
      llm_summary: llmSummaryObj.summary ?? r.summary ?? toKoreanSummary(decision, defectType, loc),

      llm_structured_json: { source: r },
      operator_note: typeof r.llm_report === "string" ? r.llm_report.trim() : "",
      action_log: toActionLog(decision, ts),
    };
  });
}
