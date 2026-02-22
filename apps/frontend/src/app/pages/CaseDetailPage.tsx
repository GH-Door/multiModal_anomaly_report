// src/app/pages/CaseDetailPage.tsx
import React, { useMemo, useState } from "react";
import { Badge } from "../components/Badge";
import {
  ChevronRight,
  Download,
  Activity,
  ShieldCheck,
  Clock,
  Loader2
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

  return (
    <div className="bg-gray-100 rounded-lg aspect-[4/3] overflow-hidden mb-4 flex items-center justify-center">
      {url ? (
        <img src={url} alt={`${active}`} className="w-full h-full object-contain" />
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
  
  // 날짜 변환 로직: 백엔드에서 온 문자열을 Date 객체로 안전하게 변환
  const formattedDate = useMemo(() => {
    const d = new Date(caseData.timestamp);
    return isNaN(d.getTime()) ? "N/A" : d.toLocaleString("ko-KR");
  }, [caseData.timestamp]);

  const llmSummary = String(caseData.llm_summary ?? "").trim();
  const isLlmComplete = llmSummary.length > 0;

  return (
    <div className="p-8 bg-white min-h-screen">
      {/* 네비게이션 */}
      <div className="flex items-center gap-2 text-sm text-gray-600 mb-6">
        <button onClick={onBackToOverview} className="hover:text-gray-900">개요</button>
        <ChevronRight className="w-4 h-4" />
        <button onClick={onBackToQueue} className="hover:text-gray-900">이상 큐</button>
        <ChevronRight className="w-4 h-4" />
        <span className="text-gray-900 font-medium">Case #{caseData.id}</span>
      </div>

      {/* 상단 헤더 정보 */}
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-gray-900 mb-2">{caseData.id}</h1>
        <p className="text-sm text-gray-500">
          {formattedDate} · {caseData.line_id}
        </p>
      </div>

      <div className="grid grid-cols-3 gap-8">
        {/* 왼쪽 섹션 (이미지 + 근거 요약) */}
        <div className="col-span-2 space-y-6">
          
          {/* 1. 검사 이미지 박스 */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-medium text-gray-900">검사 이미지</h2>
              <div className="flex items-center gap-2">
                {(["original", "heatmap", "overlay"] as const).map((k) => (
                  <button
                    key={k}
                    onClick={() => setActiveTab(k)}
                    className={`px-3 py-1.5 text-sm rounded ${
                      activeTab === k ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                    }`}
                  >
                    {k === "original" ? "원본" : k === "heatmap" ? "Heatmap" : "Overlay"}
                  </button>
                ))}
              </div>
            </div>
            <ImagePanel caseData={caseData} active={activeTab} />
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-500">이미지 ID:</span>
                <span className="ml-2 font-mono text-gray-900">{caseData.image_id}</span>
              </div>
            </div>
          </div>

          {/* 2. 근거 요약 박스 */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-6">근거 요약</h2>

            <div className="grid grid-cols-3 gap-6 mb-8">
              <div>
                <label className="text-sm font-medium text-gray-500 uppercase block mb-2">결함 타입</label>
                <p className="text-lg text-gray-900 font-semibold">{defectTypeLabel(caseData.defect_type)}</p>
              </div>
              <div>
                <label className="text-sm font-medium text-gray-500 uppercase block mb-2">위치</label>
                <p className="text-lg text-gray-900 font-semibold">{locationLabel(caseData.location)}</p>
              </div>
              <div>
                <label className="text-sm font-medium text-gray-500 uppercase block mb-2">영향 면적</label>
                <p className="text-lg text-gray-900 font-semibold">{caseData.affected_area_pct.toFixed(1)}%</p>
              </div>
            </div>

            <div className="bg-blue-50 border border-blue-100 rounded-lg p-5">
              <label className="text-sm font-bold text-blue-900 uppercase block mb-3 flex items-center gap-2">
                <Activity className={`w-4 h-4 ${!isLlmComplete ? 'animate-pulse' : ''}`} />
                AI 분석 요약
              </label>
              
              {isLlmComplete ? (
                <p className="text-sm text-blue-900 leading-relaxed whitespace-pre-wrap">
                  {llmSummary}
                </p>
              ) : (
                <div className="flex items-center gap-3 py-2">
                  <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
                  <p className="text-sm text-blue-700 font-medium italic">실시간 심층 분석 진행 중 (RAG & LLM)...</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* 오른쪽 섹션 (자율 판정 + 내보내기) */}
        <div className="space-y-6">
          
          {/* 3. 자율 시스템 판정 및 타임라인 */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
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
                    <div className={`w-1.5 h-1.5 rounded-full mt-1.5 ${isLlmComplete ? 'bg-green-500' : 'bg-gray-200'}`}></div>
                    <div>
                      <p className={`text-sm font-semibold ${isLlmComplete ? 'text-gray-900' : 'text-gray-300'}`}>
                        최종 판정 기록 완료
                      </p>
                      {isLlmComplete ? (
                        <p className="text-[11px] text-gray-500 mt-0.5">데이터베이스 동기화 성공</p>
                      ) : (
                        <div className="flex items-center gap-1.5 mt-1">
                          <span className="inline-block w-1 h-1 bg-gray-400 rounded-full animate-bounce"></span>
                          <p className="text-[11px] text-blue-500 font-medium tracking-tight">AI 분석 결과 대기 중...</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* 4. 내보내기 박스 */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">내보내기</h2>
            <button
              onClick={() => alert("리포트를 PDF 형식으로 내보냅니다.")}
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
