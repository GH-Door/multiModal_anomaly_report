// src/app/pages/CaseDetailPage.tsx
import React, { useMemo, useState, useRef } from "react";
import { Badge } from "../components/Badge";
import {
  ChevronRight,
  Download,
  Activity,
  ShieldCheck,
  Clock,
  Loader2
} from "lucide-react";
// import type { AnomalyCase } from "../data/mockData";  <- 삭제됨
import { decisionLabel, defectTypeLabel, locationLabel } from "../utils/labels";
import { getCaseImageUrl, type ImageVariant } from "../services/media";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";
import { ReportTemplate, type DBReport } from "../components/ReportTemplate";

// 외부 파일 의존성을 없애기 위해 내부에 인터페이스 정의
interface AnomalyCase {
  id: string | number;
  image_id?: string;
  timestamp: string | Date;
  line_id?: string;
  product_group?: string;
  defect_type: string;
  location: string;
  affected_area_pct?: number;
  anomaly_score?: number;
  defect_confidence?: number;
  decision: "ok" | "ng" | "pending" | "anomaly" | "normal";
  llm_summary?: string;
  llm_analysis_summary?: string;
  llm_structured_json?: any;
  pipeline_status?: string;
  pipeline_stage?: string;
  pipeline_error?: string;
  image_path?: string;
  heatmap_path?: string;
  overlay_path?: string;
}

interface CaseDetailPageProps {
  caseData: AnomalyCase;
  onBackToQueue: () => void;
  onBackToOverview: () => void;
}

function normalizeDecisionForPdf(decision?: string): DBReport["decision"] {
  const d = String(decision ?? "").trim().toLowerCase();
  if (d === "ng" || d === "anomaly") return "ng";
  if (d === "ok" || d === "normal") return "ok";
  return "pending";
}

function ImagePanel({ caseData, active }: { caseData: AnomalyCase; active: ImageVariant }) {
  const url = useMemo(() => getCaseImageUrl(caseData as any, active), [caseData, active]);

  return (
    <div className="bg-gray-100 rounded-lg aspect-[4/3] overflow-hidden mb-4 flex items-center justify-center">
      {url ? (
        <img
          src={url}
          alt={`${active}`}
          className="w-full h-full object-contain"
          loading="lazy"
          decoding="async"
        />
      ) : (
        <div className="text-gray-400 flex flex-col items-center">
          <Loader2 className="w-6 h-6 animate-spin mb-2" />
          <p className="text-xs font-medium text-gray-400">이미지 로드 중...</p>
        </div>
      )}
    </div>
  );
}

export function CaseDetailPage({ caseData, onBackToQueue, onBackToOverview }: CaseDetailPageProps) {
  const [activeTab, setActiveTab] = useState<ImageVariant>("original");
  const reportRef = useRef<HTMLDivElement>(null);
  const [isDownloading, setIsDownloading] = useState(false);

  const formattedDate = useMemo(() => {
    const d = new Date(caseData.timestamp);
    return isNaN(d.getTime()) ? "N/A" : d.toLocaleString("ko-KR");
  }, [caseData.timestamp]);

  const llmView = useMemo(() => {
    const source = (caseData.llm_structured_json as any)?.source ?? {};

    const asObj = (v: any) => {
      if (!v) return {};
      if (typeof v === "object") return v;
      if (typeof v === "string") {
        const s = v.trim();
        if (!s) return {};
        try {
          return JSON.parse(s);
        } catch {
          return { _text: s };
        }
      }
      return {};
    };
    const pick = (...values: any[]) => {
      for (const v of values) {
        if (typeof v === "string" && v.trim()) return v.trim();
      }
      return "";
    };

    const llmReportRoot = asObj(source.llm_report);
    const llmReport = asObj(llmReportRoot.report ?? llmReportRoot);
    const llmSummaryObj = asObj(source.llm_summary);

    const summary = pick(
      caseData.llm_summary,
      llmSummaryObj.summary,
      llmSummaryObj._text,
      llmReport.summary,
      source.summary
    );
    const description = pick(
      llmReport.description,
      source.defect_description,
      source.description
    );
    const cause = pick(
      llmReport.possible_cause,
      llmReport.cause,
      llmReport.likely_cause,
      source.possible_cause
    );
    const recommendation = pick(
      llmReport.recommendation,
      source.recommendation
    );
    const pipelineStatus = String(
      source.pipeline_status ?? caseData.pipeline_status ?? ""
    ).toLowerCase();
    const pipelineStage = String(source.pipeline_stage ?? caseData.pipeline_stage ?? "");
    const pipelineError = pick(source.pipeline_error, caseData.pipeline_error);
    return { summary, description, cause, recommendation, pipelineStatus, pipelineStage, pipelineError };
  }, [caseData.llm_structured_json, caseData.llm_summary]);

  const isLlmComplete =
    llmView.summary.length > 0 ||
    llmView.description.length > 0 ||
    llmView.cause.length > 0 ||
    llmView.recommendation.length > 0;
  const isLlmProcessing = llmView.pipelineStatus === "processing";
  const isLlmFailed = llmView.pipelineStatus === "failed";
  const llmFallbackText = isLlmFailed
    ? `LLM 분석 실패${llmView.pipelineError ? `: ${llmView.pipelineError}` : ""}`
    : isLlmProcessing
      ? "LLM 분석 진행 중..."
      : "LLM 분석 결과가 아직 없습니다. (생성 실패 또는 미완료)";
  
  const llmSummaryForPdf = isLlmComplete
    ? [
        llmView.summary,
        llmView.description ? `상세 분석: ${llmView.description}` : "",
        llmView.cause ? `원인 추정: ${llmView.cause}` : "",
        llmView.recommendation ? `권고 조치: ${llmView.recommendation}` : "",
      ]
        .filter((v) => v.trim().length > 0)
        .join("\n")
    : llmFallbackText;

  const pdfReportData = useMemo<DBReport>(() => {
    const timestamp =
      caseData.timestamp instanceof Date
        ? caseData.timestamp.toISOString()
        : String(caseData.timestamp ?? "");

    return {
      id: caseData.id,
      image_id: caseData.image_id ?? "",
      category: caseData.product_group ?? "-",
      timestamp,
      defect_type: caseData.defect_type ?? "",
      location: caseData.location ?? "",
      affected_area_pct:
        typeof caseData.affected_area_pct === "number" ? caseData.affected_area_pct : 0,
      ad_score: typeof caseData.anomaly_score === "number" ? caseData.anomaly_score : 0,
      confidence: typeof caseData.defect_confidence === "number" ? caseData.defect_confidence : 0,
      decision: normalizeDecisionForPdf(caseData.decision),
      llm_analysis_summary: llmSummaryForPdf,
      image_path: caseData.image_path ?? "",
      heatmap_path: caseData.heatmap_path ?? "",
      overlay_path: caseData.overlay_path ?? "",
    };
  }, [caseData, llmSummaryForPdf]);

  const waitForReportImages = async (root: HTMLElement) => {
    const images = Array.from(root.querySelectorAll("img"));
    await Promise.all(
      images.map((img) => {
        if (img.complete) return Promise.resolve();
        return new Promise<void>((resolve) => {
          const done = () => resolve();
          img.addEventListener("load", done, { once: true });
          img.addEventListener("error", done, { once: true });
        });
      })
    );
  };

  const handleDownloadPdf = async () => {
    if (!reportRef.current) return;

    setIsDownloading(true);
    try {
      const element = reportRef.current;
      await waitForReportImages(element);

      const canvas = await html2canvas(element, {
        scale: 2,
        useCORS: true,
        imageTimeout: 15000,
        logging: false,
        backgroundColor: "#ffffff",
        removeContainer: true,
        onclone: (clonedDoc) => {
          const allElements = clonedDoc.getElementsByTagName("*");
          for (let i = 0; i < allElements.length; i++) {
            const el = allElements[i] as HTMLElement;
            const style = window.getComputedStyle(el);
            if (style.color?.includes("oklch")) el.style.color = "#111827";
            if (style.backgroundColor?.includes("oklch")) el.style.backgroundColor = "#ffffff";
            if (style.borderColor?.includes("oklch")) el.style.borderColor = "#e5e7eb";
          }
        }
      });

      const imgData = canvas.toDataURL("image/png");
      const pdf = new jsPDF("p", "mm", "a4");
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const imgProps = pdf.getImageProperties(imgData);
      const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;

      pdf.addImage(imgData, "PNG", 0, 0, pdfWidth, pdfHeight);
      pdf.save(`Inspection_Report_${caseData.image_id || caseData.id}.pdf`);
      
    } catch (error) {
      console.error("PDF 생성 실패:", error);
      alert("PDF 생성 중 오류가 발생했습니다.");
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className="p-8 bg-white min-h-screen">
      <div className="flex items-center gap-2 text-sm text-gray-600 mb-6">
        <button onClick={onBackToOverview} className="hover:text-gray-900">개요</button>
        <ChevronRight className="w-4 h-4" />
        <button onClick={onBackToQueue} className="hover:text-gray-900">이상 큐</button>
        <ChevronRight className="w-4 h-4" />
        <span className="text-gray-900 font-medium">Case #{caseData.id}</span>
      </div>

      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-gray-900 mb-2">Case #{caseData.id}</h1>
        <p className="text-sm text-gray-500">
          {formattedDate} · {caseData.line_id}
        </p>
      </div>

      <div className="grid grid-cols-3 gap-8">
        <div className="col-span-2 space-y-6">
          <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-medium text-gray-900">검사 이미지</h2>
              <div className="flex items-center gap-2">
                {(["original", "heatmap", "overlay"] as const).map((k) => (
                  <button
                    key={k}
                    onClick={() => setActiveTab(k)}
                    className={`px-3 py-1.5 text-sm rounded transition-colors ${
                      activeTab === k ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                    }`}
                  >
                    {k === "original" ? "원본" : k === "heatmap" ? "Heatmap" : "Overlay"}
                  </button>
                ))}
              </div>
            </div>
            <ImagePanel caseData={caseData} active={activeTab} />
            <div className="text-sm">
              <span className="text-gray-500 font-medium">이미지 ID:</span>
              <span className="ml-2 font-mono text-gray-900">{caseData.image_id}</span>
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
            <h2 className="text-lg font-medium text-gray-900 mb-6">근거 요약</h2>
            <div className="grid grid-cols-3 gap-6 mb-8">
              <div>
                <label className="text-xs font-bold text-gray-400 uppercase block mb-1">결함 타입</label>
                <p className="text-lg text-gray-900 font-semibold">{defectTypeLabel(caseData.defect_type)}</p>
              </div>
              <div>
                <label className="text-xs font-bold text-gray-400 uppercase block mb-1">위치</label>
                <p className="text-lg text-gray-900 font-semibold">{locationLabel(caseData.location)}</p>
              </div>
              <div>
                <label className="text-xs font-bold text-gray-400 uppercase block mb-1">영향 면적</label>
                <p className="text-lg text-gray-900 font-semibold">{caseData.affected_area_pct?.toFixed(1)}%</p>
              </div>
            </div>

            <div className="bg-blue-50 border border-blue-100 rounded-lg p-5">
              <label className="text-sm font-bold text-blue-900 uppercase block mb-3 flex items-center gap-2">
                <Activity className={`w-4 h-4 ${!isLlmComplete ? 'animate-pulse' : ''}`} />
                AI 분석 요약
              </label>
              {isLlmComplete ? (
                <div className="space-y-3 text-sm text-blue-900 leading-relaxed">
                  <p className="whitespace-pre-wrap">{llmView.summary}</p>
                  {llmView.description && (
                    <p><span className="font-semibold">상세 분석:</span> {llmView.description}</p>
                  )}
                  {llmView.cause && (
                    <p><span className="font-semibold">원인 추정:</span> {llmView.cause}</p>
                  )}
                  {llmView.recommendation && (
                    <p><span className="font-semibold">권고 조치:</span> {llmView.recommendation}</p>
                  )}
                </div>
              ) : (
                <div className="flex items-center gap-3 py-2">
                  <Loader2 className={`w-4 h-4 text-blue-500 ${isLlmProcessing ? "animate-spin" : ""}`} />
                  <p className="text-sm text-blue-700 font-medium italic">
                    {llmFallbackText}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-medium text-gray-900 flex items-center gap-2">
                <ShieldCheck className="w-5 h-5 text-green-600" />
                자율 시스템 판정
              </h2>
              <Badge variant={caseData.decision} className="text-sm px-3 py-1">
                {decisionLabel(caseData.decision)}
              </Badge>
            </div>
            <div className="space-y-6">
              <div className="bg-gray-50 rounded-lg p-4 border border-gray-100">
                <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                  <Clock className="w-3 h-3" />
                  Processing History
                </h3>
                
                <div className="space-y-5">
                  <div className="flex gap-3">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-1.5 shadow-[0_0_8px_rgba(59,130,246,0.3)]"></div>
                    <div>
                      <p className="text-sm font-semibold text-gray-900">AD 분석 완료</p>
                      <p className="text-[11px] text-gray-500 mt-0.5">{formattedDate}</p>
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <div className={`w-1.5 h-1.5 rounded-full mt-1.5 ${
                      isLlmComplete ? 'bg-green-500' : isLlmFailed ? 'bg-red-400' : 'bg-blue-400'
                    }`}></div>
                    <div>
                      <p className={`text-sm font-semibold ${
                        isLlmComplete ? 'text-gray-900' : isLlmFailed ? 'text-red-600' : 'text-blue-700'
                      }`}>
                        {isLlmComplete ? "최종 판정 기록 완료" : isLlmFailed ? "LLM 분석 실패" : "LLM 분석 진행 중"}
                      </p>
                      {isLlmComplete ? (
                        <p className="text-[11px] text-gray-500 mt-0.5">데이터베이스 동기화 성공</p>
                      ) : (
                        <div className="flex items-center gap-1.5 mt-1">
                          {!isLlmFailed && <span className="inline-block w-1 h-1 bg-blue-400 rounded-full animate-bounce"></span>}
                          <p className={`text-[11px] font-medium tracking-tight ${isLlmFailed ? "text-red-500" : "text-blue-500"}`}>
                            {isLlmFailed
                              ? (llmView.pipelineError || "로그를 확인해주세요.")
                              : (llmView.pipelineStage || "AI 분석 결과 대기 중...")}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
            <h2 className="text-lg font-medium text-gray-900 mb-4">내보내기</h2>
            <button
              onClick={handleDownloadPdf}
              disabled={isDownloading}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:bg-blue-300"
            >
              {isDownloading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
              <span>{isDownloading ? "생성 중..." : "PDF 다운로드"}</span>
            </button>
          </div>
        </div>
      </div>

      <div style={{ 
        position: 'absolute', 
        top: '-10000px', 
        left: '-10000px',
        all: 'initial' 
      }}>
        <div ref={reportRef}>
          <ReportTemplate reportData={pdfReportData} />
        </div>
      </div>
    </div>
  );
}