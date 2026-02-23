// src/app/components/ReportTemplate.tsx
import React, { forwardRef } from "react";
import { defectTypeLabel, locationLabel } from "../utils/labels";
import { getCaseImageUrl } from "../services/media";

export interface DBReport {
  id: number;
  image_id: string;
  category: string;
  timestamp: string;
  defect_type: string;
  location: string;
  affected_area_pct: number;
  ad_score: number;
  confidence: number;
  decision: "ok" | "ng" | "pending" | "anomaly" | "normal";
  llm_analysis_summary: string;
  image_path: string;
  heatmap_path: string;
  overlay_path: string;
}

interface ReportTemplateProps {
  reportData: DBReport;
}

export const ReportTemplate = forwardRef<HTMLDivElement, ReportTemplateProps>(
  ({ reportData }, ref) => {
    const formattedDate = new Date(reportData.timestamp).toLocaleString("ko-KR");
    const originalUrl = getCaseImageUrl(reportData as any, "original");
    const heatmapUrl = getCaseImageUrl(reportData as any, "heatmap");

    // 판정 텍스트 및 스타일 결정
    const isAnomaly = reportData.decision === "ng" || reportData.decision === "anomaly";

    return (
      /* PDF 렌더링을 위한 컨테이너 - 화면 밖으로 숨겨두고 PDF 생성 시에만 참조함 */
      <div 
        ref={ref}
        className="p-10 bg-white font-sans text-gray-900 w-[800px]" // PDF 규격에 맞춘 고정폭
      >
        {/* 헤더 섹션 */}
        <div className="bg-[#1e3a5f] text-white p-4 flex justify-between items-center">
          <h1 className="text-xl font-bold tracking-tight">MMAD INSPECTOR</h1>
          <h2 className="text-xl font-bold">단일 이미지 검사 리포트</h2>
        </div>

        {/* 요약 테이블 */}
        <div className="grid grid-cols-4 border-x border-b border-blue-200 text-center text-sm mb-8">
          <div className="border-r border-blue-100 p-3 bg-blue-50">
            <p className="text-xs text-gray-500 mb-1">리포트 번호</p>
            <p className="font-bold"># {reportData.id}</p>
          </div>
          <div className="border-r border-blue-100 p-3 bg-blue-50">
            <p className="text-xs text-gray-500 mb-1">데이터셋</p>
            <p className="font-bold">GoodsAD</p>
          </div>
          <div className="border-r border-blue-100 p-3 bg-blue-50">
            <p className="text-xs text-gray-500 mb-1">제품 카테고리</p>
            <p className="font-bold">{reportData.category}</p>
          </div>
          <div className="p-3 bg-blue-50">
            <p className="text-xs text-gray-500 mb-1">검사 일시</p>
            <p className="font-bold text-[11px]">{formattedDate}</p>
          </div>
        </div>

        {/* 이미지 분석 */}
        <section className="mb-8">
          <h3 className="text-lg font-bold border-b-2 border-gray-400 pb-2 mb-4 text-[#1e3a5f]">이미지 분석</h3>
          <div className="grid grid-cols-2 gap-6">
            <div className="text-center">
              <div className="bg-gray-100 aspect-square rounded border border-gray-200 mb-2 flex items-center justify-center overflow-hidden">
                {originalUrl ? <img src={originalUrl} crossOrigin="anonymous" className="w-full h-full object-contain" /> : "No Image"}
              </div>
              <p className="text-sm font-medium text-gray-600">원본 이미지</p>
            </div>
            <div className="text-center">
              <div className="bg-gray-100 aspect-square rounded border border-gray-200 mb-2 flex items-center justify-center overflow-hidden">
                {heatmapUrl ? <img src={heatmapUrl} crossOrigin="anonymous" className="w-full h-full object-contain" /> : "No Heatmap"}
              </div>
              <p className="text-sm font-medium text-gray-600">이상 히트맵 (Score: {reportData.ad_score?.toFixed(2)})</p>
            </div>
          </div>
        </section>

        {/* 검사 판정 */}
        <section className="mb-8">
          <h3 className="text-lg font-bold border-b-2 border-gray-400 pb-2 mb-4 text-[#1e3a5f]">검사 판정</h3>
          <div className="grid grid-cols-3 bg-[#1e3a5f] text-white text-center text-sm py-2">
            <div>AD 분석 결과</div>
            <div>VLM 판정</div>
            <div>최종 판정</div>
          </div>
          <div className="grid grid-cols-3 border-x border-b border-gray-200 text-center py-4 font-bold text-lg">
            <div className={isAnomaly ? "text-red-500" : "text-green-500"}>{isAnomaly ? "불량" : "정상"}</div>
            <div className={isAnomaly ? "text-red-500" : "text-green-500"}>{isAnomaly ? "불량" : "정상"}</div>
            <div className={`py-1 ${isAnomaly ? "text-red-600 bg-red-50" : "text-green-600 bg-green-50"}`}>
              {isAnomaly ? "출하 불가" : "출하 가능"}
            </div>
          </div>
        </section>

        {/* 결함 상세 분석 */}
        <section className="mb-8">
          <h3 className="text-lg font-bold border-b-2 border-gray-400 pb-2 mb-4 text-[#1e3a5f]">결함 상세 분석</h3>
          <table className="w-full border-collapse border border-gray-200 text-sm">
            <tbody>
              <tr>
                <td className="w-1/4 bg-gray-50 p-3 border border-gray-200 font-bold text-center">결함 유형</td>
                <td className="p-3 border border-gray-200">{defectTypeLabel(reportData.defect_type)}</td>
              </tr>
              <tr>
                <td className="bg-gray-50 p-3 border border-gray-200 font-bold text-center">결함 위치</td>
                <td className="p-3 border border-gray-200">{locationLabel(reportData.location)}</td>
              </tr>
              <tr>
                <td className="bg-gray-50 p-3 border border-gray-200 font-bold text-center">상세 설명</td>
                <td className="p-3 border border-gray-200 leading-relaxed text-gray-700 min-h-[80px]">
                  {reportData.llm_analysis_summary}
                </td>
              </tr>
              <tr>
                <td className="bg-gray-50 p-3 border border-gray-200 font-bold text-center">신뢰도</td>
                <td className="p-3 border border-gray-200">{(reportData.confidence * 100).toFixed(0)}%</td>
              </tr>
            </tbody>
          </table>
        </section>

        {/* 하단 조치 사항 */}
        {isAnomaly && (
          <div className="bg-red-50 border border-red-200 p-4 rounded mt-4">
            <h4 className="text-sm font-bold text-red-800 mb-1">조치 권고사항</h4>
            <p className="text-sm text-red-700">즉시 생산 라인에서 제거 후 불량 처리 및 정밀 재검수 필요</p>
          </div>
        )}
      </div>
    );
  }
);