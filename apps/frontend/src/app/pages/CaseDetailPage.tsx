// src/app/pages/CaseDetailPage.tsx
import React, { useMemo, useState } from "react";
import { Badge } from "../components/Badge";
import {
  ChevronRight,
  Download,
  MessageSquare,
  CheckCircle,
  XCircle,
  RotateCcw,
  FileText,
} from "lucide-react";
import type { AnomalyCase } from "../data/mockData";
import { decisionLabel, defectTypeLabel, locationLabel } from "../utils/labels";
import { getCaseImageUrl, type ImageVariant } from "../services/media";

interface CaseDetailPageProps {
  caseData: AnomalyCase;
  onBackToQueue: () => void;
  onBackToOverview: () => void;
}

function ImagePanel({ caseData, active }: { caseData: AnomalyCase; active: ImageVariant }) {
  const url = useMemo(() => getCaseImageUrl(caseData, active), [caseData, active]);

  if (url) {
    return (
      <div className="bg-gray-100 rounded-lg aspect-[4/3] overflow-hidden mb-4">
        <img src={url} alt={`${active}`} className="w-full h-full object-contain bg-black/5" />
      </div>
    );
  }

  return (
    <div className="bg-gray-100 rounded-lg aspect-[4/3] flex items-center justify-center mb-4">
      <div className="text-center">
        <div className="w-24 h-24 bg-gray-300 rounded-lg mx-auto mb-3" />
        <p className="font-medium text-gray-700">이미지 {active}</p>
        <p className="text-sm text-gray-500 mt-1">{caseData.image_id}</p>
        <p className="text-xs text-gray-400 mt-2">* 백엔드에서 경로가 제공되면 자동으로 표시됩니다.</p>
      </div>
    </div>
  );
}

export function CaseDetailPage({ caseData, onBackToQueue, onBackToOverview }: CaseDetailPageProps) {
  const [activeTab, setActiveTab] = useState<ImageVariant>("original");
  const [showJson, setShowJson] = useState(false);
  const [note, setNote] = useState("");

  const handleAction = (action: string) => {
    if (action === "저장") return alert("메모를 저장합니다.");
    if (action === "PDF") return alert("리포트를 PDF 형식으로 내보냅니다.");
    alert(`${action} 되었습니다.`);
  };

  return (
    <div className="p-8">
      <div className="flex items-center gap-2 text-sm text-gray-600 mb-6">
        <button onClick={onBackToOverview} className="hover:text-gray-900">
          개요
        </button>
        <ChevronRight className="w-4 h-4" />
        <button onClick={onBackToQueue} className="hover:text-gray-900">
          이상 큐
        </button>
        <ChevronRight className="w-4 h-4" />
        <span className="text-gray-900 font-medium">케이스 상세</span>
      </div>

      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900 mb-2">{caseData.id}</h1>
          <p className="text-sm text-gray-600">
            {caseData.timestamp.toLocaleString("ko-KR")} · {caseData.line_id} · {caseData.shift}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Badge variant={caseData.decision}>{decisionLabel(caseData.decision)}</Badge>
          <Badge variant={caseData.severity}>
            {caseData.severity === "high" ? "높음" : caseData.severity === "med" ? "중간" : "낮음"}
          </Badge>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6">
        <div className="col-span-2">
          <div className="bg-white border border-gray-200 rounded-lg p-6 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-medium text-gray-900">검사 이미지</h2>
              <div className="flex items-center gap-2">
                {(["original", "heatmap", "overlay"] as const).map((k) => (
                  <button
                    key={k}
                    onClick={() => setActiveTab(k)}
                    className={`px-3 py-1.5 text-sm rounded ${
                      activeTab === k
                        ? "bg-blue-600 text-white"
                        : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                    }`}
                  >
                    {k === "original" ? "원본" : k === "heatmap" ? "Heatmap" : "Overlay"}
                  </button>
                ))}
              </div>
            </div>

            <ImagePanel caseData={caseData} active={activeTab} />

            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <span className="text-gray-500">이미지 ID:</span>
                <span className="ml-2 font-mono text-gray-900">{caseData.image_id}</span>
              </div>
              <div>
                <span className="text-gray-500">영향 영역:</span>
                <span className="ml-2 font-medium text-gray-900">{caseData.affected_area_pct.toFixed(1)}%</span>
              </div>
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-6 mb-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">근거 요약</h2>

            <div className="mb-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700">Anomaly Score</span>
                <span className="text-2xl font-semibold text-gray-900">{caseData.anomaly_score.toFixed(3)}</span>
              </div>

              <div className="h-4 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className={`h-full ${
                    caseData.anomaly_score >= 0.8
                      ? "bg-red-500"
                      : caseData.anomaly_score >= 0.65
                        ? "bg-orange-500"
                        : "bg-green-500"
                  }`}
                  style={{ width: `${caseData.anomaly_score * 100}%` }}
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-6 mb-6">
              <div>
                <label className="text-sm font-medium text-gray-500 uppercase block mb-2">결함 타입</label>
                <p className="text-lg text-gray-900">{defectTypeLabel(caseData.defect_type)}</p>
                <p className="text-sm text-gray-500 mt-1">신뢰도: {(caseData.defect_confidence * 100).toFixed(1)}%</p>
              </div>

              <div>
                <label className="text-sm font-medium text-gray-500 uppercase block mb-2">위치</label>
                <p className="text-lg text-gray-900">{locationLabel(caseData.location)}</p>
                <p className="text-sm text-gray-500 mt-1">영향 면적: {caseData.affected_area_pct.toFixed(1)}%</p>
              </div>
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <label className="text-sm font-medium text-blue-900 uppercase block mb-2">AI 분석 요약</label>
              <p className="text-sm text-blue-900">{caseData.llm_summary}</p>
            </div>

            <div className="mt-4">
              <button
                onClick={() => setShowJson(!showJson)}
                className="text-sm text-blue-600 hover:text-blue-700 flex items-center gap-1"
              >
                <FileText className="w-4 h-4" />
                {showJson ? "구조화된 분석 결과 숨기기" : "구조화된 분석 결과 보기"}
              </button>

              {showJson && (
                <pre className="mt-3 p-4 bg-gray-900 text-green-400 rounded-lg text-xs overflow-x-auto">
                  {JSON.stringify(caseData.llm_structured_json, null, 2)}
                </pre>
              )}
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">작업 로그</h2>
            <div className="space-y-3">
              {caseData.action_log.map((log, idx) => (
                <div
                  key={idx}
                  className="flex items-start gap-3 pb-3 border-b border-gray-100 last:border-0 last:pb-0"
                >
                  <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-sm font-medium text-blue-700">{log.who[0]}</span>
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-medium text-gray-900">{log.what}</p>
                    <p className="text-xs text-gray-500 mt-1">
                      {log.who} · {log.when.toLocaleString("ko-KR")}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">처리</h2>
            <div className="space-y-2">
              <button
                onClick={() => handleAction("승인")}
                className="w-full flex items-center gap-2 px-4 py-2.5 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
              >
                <CheckCircle className="w-4 h-4" />
                <span>승인</span>
              </button>
              <button
                onClick={() => handleAction("불량 확정")}
                className="w-full flex items-center gap-2 px-4 py-2.5 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                <XCircle className="w-4 h-4" />
                <span>불량 확정</span>
              </button>
              <button
                onClick={() => handleAction("재검 요청")}
                className="w-full flex items-center gap-2 px-4 py-2.5 bg-amber-600 text-white rounded-lg hover:bg-amber-700 transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                <span>재검 요청</span>
              </button>
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">메모 추가</h2>
            <textarea
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="케이스에 대한 메모를 입력하세요."
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
              rows={4}
            />
            <button
              onClick={() => {
                handleAction("저장");
                setNote("");
              }}
              className="mt-3 w-full flex items-center justify-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
            >
              <MessageSquare className="w-4 h-4" />
              <span>메모 저장</span>
            </button>

            {caseData.operator_note && (
              <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                <p className="text-xs font-medium text-gray-500 mb-1">기존 메모</p>
                <p className="text-sm text-gray-700">{caseData.operator_note}</p>
              </div>
            )}
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">내보내기</h2>
            <button
              onClick={() => handleAction("PDF")}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
            >
              <Download className="w-4 h-4" />
              <span>PDF 다운로드</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
