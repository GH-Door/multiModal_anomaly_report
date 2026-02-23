// src/app/components/ReportTemplate.tsx
import React, { forwardRef } from "react";
import { defectTypeLabel, locationLabel } from "../utils/labels";
import { getCaseImageUrl } from "../services/media";

export interface DBReport {
  id: string | number;
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
    const rawDate = new Date(reportData.timestamp);
    const formattedDate = isNaN(rawDate.getTime()) ? "-" : rawDate.toLocaleString("ko-KR");
    const originalUrl = getCaseImageUrl(reportData as any, "original");
    const heatmapUrl = getCaseImageUrl(reportData as any, "heatmap");

    const decision = String(reportData.decision ?? "").trim().toLowerCase();
    const isAnomaly = decision === "ng" || decision === "anomaly";
    const adScoreText = Number.isFinite(reportData.ad_score) ? reportData.ad_score.toFixed(2) : "-";
    const confidenceText = Number.isFinite(reportData.confidence)
      ? `${Math.round(reportData.confidence * 100)}%`
      : "-";
    const llmSummaryText = String(reportData.llm_analysis_summary ?? "").trim() || "-";

    // ✅ 모든 컬러를 oklch가 아닌 HEX(#)로 정의
    const styles = {
      container: { backgroundColor: '#ffffff', color: '#111827', width: '794px', padding: '40px', fontFamily: 'sans-serif', textAlign: 'left' as const },
      header: { backgroundColor: '#1e3a5f', color: '#ffffff', padding: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' },
      sectionTitle: { fontSize: '18px', fontWeight: 'bold', borderBottom: '2px solid #9ca3af', paddingBottom: '8px', marginBottom: '16px', color: '#1e3a5f', marginTop: '24px' },
      summaryBox: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', border: '1px solid #dbeafe', textAlign: 'center' as const, fontSize: '13px', marginBottom: '30px' },
      cellLabel: { backgroundColor: '#f9fafb', padding: '12px', border: '1px solid #e5e7eb', fontWeight: 'bold', textAlign: 'center' as const, fontSize: '14px', color: '#374151' },
      cellContent: { padding: '12px', border: '1px solid #e5e7eb', fontSize: '14px', color: '#111827' }
    };

    return (
      <div ref={ref} style={styles.container}>
        <div style={styles.header}>
          <h1 style={{ fontSize: '20px', fontWeight: 'bold', margin: 0 }}>MMAD INSPECTOR</h1>
          <h2 style={{ fontSize: '16px', margin: 0 }}>품질 검사 결과 리포트</h2>
        </div>

        <div style={styles.summaryBox}>
          <div style={{ backgroundColor: '#eff6ff', padding: '12px', borderRight: '1px solid #dbeafe' }}>
            <p style={{ color: '#6b7280', margin: '0 0 4px 0', fontSize: '11px' }}>리포트 ID</p>
            <p style={{ fontWeight: 'bold', margin: 0 }}># {reportData.id}</p>
          </div>
          <div style={{ backgroundColor: '#eff6ff', padding: '12px', borderRight: '1px solid #dbeafe' }}>
            <p style={{ color: '#6b7280', margin: '0 0 4px 0', fontSize: '11px' }}>카테고리</p>
            <p style={{ fontWeight: 'bold', margin: 0 }}>{reportData.category || "-"}</p>
          </div>
          <div style={{ backgroundColor: '#eff6ff', padding: '12px', borderRight: '1px solid #dbeafe' }}>
            <p style={{ color: '#6b7280', margin: '0 0 4px 0', fontSize: '11px' }}>AD 점수</p>
            <p style={{ fontWeight: 'bold', margin: 0 }}>{adScoreText}</p>
          </div>
          <div style={{ backgroundColor: '#eff6ff', padding: '12px' }}>
            <p style={{ color: '#6b7280', margin: '0 0 4px 0', fontSize: '11px' }}>검사 시간</p>
            <p style={{ fontWeight: 'bold', margin: 0, fontSize: '10px' }}>{formattedDate}</p>
          </div>
        </div>

        <section>
          <h3 style={styles.sectionTitle}>이미지 분석 데이터</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ backgroundColor: '#f3f4f6', height: '300px', border: '1px solid #e5e7eb', marginBottom: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {originalUrl ? <img src={originalUrl} crossOrigin="anonymous" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} /> : "No Image"}
              </div>
              <p style={{ fontSize: '13px', color: '#4b5563' }}>원본 이미지</p>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ backgroundColor: '#f3f4f6', height: '300px', border: '1px solid #e5e7eb', marginBottom: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {heatmapUrl ? <img src={heatmapUrl} crossOrigin="anonymous" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} /> : "No Heatmap"}
              </div>
              <p style={{ fontSize: '13px', color: '#4b5563' }}>이상 영역 히트맵</p>
            </div>
          </div>
        </section>

        <section style={{ marginTop: '30px' }}>
          <h3 style={styles.sectionTitle}>판정 결과</h3>
          <div style={{ display: 'flex', border: '2px solid #1e3a5f', textAlign: 'center' as const, fontSize: '18px', fontWeight: 'bold' }}>
            <div style={{ flex: 1, padding: '15px', borderRight: '2px solid #1e3a5f' }}>
              시스템 판정: <span style={{ color: isAnomaly ? '#ef4444' : '#10b981' }}>{isAnomaly ? "불량(NG)" : "정상(OK)"}</span>
            </div>
            <div style={{ flex: 1, padding: '15px', backgroundColor: isAnomaly ? '#fef2f2' : '#f0fdf4', color: isAnomaly ? '#dc2626' : '#059669' }}>
              {isAnomaly ? "출하 불가" : "출하 승인"}
            </div>
          </div>
        </section>

        <section style={{ marginTop: '30px' }}>
          <h3 style={styles.sectionTitle}>상세 분석 내역</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', border: '1px solid #e5e7eb' }}>
            <tbody>
              <tr>
                <td style={styles.cellLabel}>결함 유형</td>
                <td style={styles.cellContent}>{defectTypeLabel(reportData.defect_type)}</td>
              </tr>
              <tr>
                <td style={styles.cellLabel}>결함 위치</td>
                <td style={styles.cellContent}>{locationLabel(reportData.location)}</td>
              </tr>
              <tr>
                <td style={styles.cellLabel}>AI 요약</td>
                <td style={{ ...styles.cellContent, lineHeight: '1.6' }}>{llmSummaryText}</td>
              </tr>
              <tr>
                <td style={styles.cellLabel}>신뢰 점수</td>
                <td style={styles.cellContent}>{confidenceText}</td>
              </tr>
            </tbody>
          </table>
        </section>
      </div>
    );
  }
);