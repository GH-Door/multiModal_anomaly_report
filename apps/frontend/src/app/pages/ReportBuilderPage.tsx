import React, { useMemo, useRef, useState } from "react";
import { Download, Loader2 } from "lucide-react";
import { domToPng } from 'modern-screenshot';
import jsPDF from 'jspdf';
import type { AnomalyCase } from "../data/mockData";
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import {
  buildAggregates, toDefectRateTrend, toDefectTypePie, toProductDefectsFixed10,
} from "../selectors/caseSelectors";

interface ReportBuilderPageProps {
  cases: AnomalyCase[];
}

export function ReportBuilderPage({ cases }: ReportBuilderPageProps) {
  const pdfRef = useRef<HTMLDivElement>(null);
  const [isDownloading, setIsDownloading] = useState(false);

  // --- 데이터 가공 로직 ---
  const agg = useMemo(() => buildAggregates(cases), [cases]);
  const reportMetrics = useMemo(() => {
    const total = agg.total;
    return {
      total,
      ngCount: agg.ng,
      reviewCount: agg.review,
      okCount: agg.ok,
      ngRate: total ? ((agg.ng / total) * 100).toFixed(2) : "0.00",
      reviewRate: total ? ((agg.review / total) * 100).toFixed(2) : "0.00",
      passRate: total ? ((agg.ok / total) * 100).toFixed(2) : "0.00",
      avgScore: total ? agg.avgScore.toFixed(3) : "0.000",
      avgInferenceTime: total ? agg.avgInference.toFixed(1) : "0.0",
    };
  }, [agg]);

  const defectRateTrend = useMemo(() => toDefectRateTrend(agg), [agg]);
  const defectTypeData = useMemo(() => toDefectTypePie(agg), [agg]);
  const productDefectsFixed10 = useMemo(() => toProductDefectsFixed10(agg), [agg]);
  const COLORS = ["#ef4444", "#f97316", "#eab308", "#84cc16", "#06b6d4", "#8b5cf6"];

  // --- PDF 생성 핸들러 ---
  const handleExportPdf = async () => {
    if (!pdfRef.current) return;
    setIsDownloading(true);

    try {
      // 보이지 않는 PDF 전용 컨테이너를 캡처
      const dataUrl = await domToPng(pdfRef.current, {
        scale: 2,
        backgroundColor: '#ffffff',
      });

      const pdf = new jsPDF("p", "mm", "a4");
      const pdfWidth = pdf.internal.pageSize.getWidth();
      
      const imgProps = pdf.getImageProperties(dataUrl);
      const imgHeight = (imgProps.height * pdfWidth) / imgProps.width;

      pdf.addImage(dataUrl, 'PNG', 0, 0, pdfWidth, imgHeight);
      pdf.save(`Operation_Report_${new Date().toISOString().split('T')[0]}.pdf`);

    } catch (error) {
      console.error("PDF 생성 에러:", error);
      alert("리포트 생성 중 오류가 발생했습니다.");
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <>
      {/* =========================================================
          1. 웹 사용자 UI (원상복구된 오리지널 레이아웃)
          ========================================================= */}
      <div className="p-8 bg-gray-50">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold text-gray-900 mb-2">운영 리포트</h1>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-6 mb-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4">요약 (Executive Summary)</h2>
          <div className="grid grid-cols-5 gap-6">
            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-sm text-gray-600 mb-1">총 검사 수</p>
              <p className="text-3xl font-semibold text-gray-900">{reportMetrics.total}</p>
            </div>
            <div className="bg-red-50 rounded-lg p-4">
              <p className="text-sm text-red-700 mb-1">불량 (NG)</p>
              <p className="text-3xl font-semibold text-red-700">{reportMetrics.ngCount}</p>
              <p className="text-xs text-red-600 mt-1">{reportMetrics.ngRate}%</p>
            </div>
            <div className="bg-amber-50 rounded-lg p-4">
              <p className="text-sm text-amber-700 mb-1">재검토 (REVIEW)</p>
              <p className="text-3xl font-semibold text-amber-700">{reportMetrics.reviewCount}</p>
              <p className="text-xs text-amber-600 mt-1">{reportMetrics.reviewRate}%</p>
            </div>
            <div className="bg-green-50 rounded-lg p-4">
              <p className="text-sm text-green-700 mb-1">정상 (OK)</p>
              <p className="text-3xl font-semibold text-green-700">{reportMetrics.okCount}</p>
              <p className="text-xs text-green-600 mt-1">{reportMetrics.passRate}%</p>
            </div>
            <div className="bg-blue-50 rounded-lg p-4">
              <p className="text-sm text-blue-700 mb-1">평균 Score</p>
              <p className="text-3xl font-semibold text-blue-700">{reportMetrics.avgScore}</p>
              <p className="text-xs text-blue-600 mt-1">{reportMetrics.avgInferenceTime}ms</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h3 className="text-lg font-bold text-gray-900 mb-4">불량률 추이</h3>
            <div className="h-[260px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={defectRateTrend}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="time" tick={{ fontSize: 12 }} stroke="#6b7280" />
                  <YAxis tick={{ fontSize: 12 }} stroke="#6b7280" />
                  <Tooltip />
                  <Line type="monotone" dataKey="불량률" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h3 className="text-lg font-bold text-gray-900 mb-4">결함 타입 분포</h3>
            <div className="h-[260px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={defectTypeData}
                    cx="50%" cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={86}
                    dataKey="value"
                  >
                    {defectTypeData.map((_, index) => <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-6 mb-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4">제품별 불량 비교</h3>
          <div className="h-[320px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={productDefectsFixed10} margin={{ bottom: 24 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} stroke="#6b7280" interval={0} angle={-15} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 12 }} stroke="#6b7280" allowDecimals={false} />
                <Tooltip />
                <Legend />
                <Bar dataKey="count" name="불량 건수(NG)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4">리포트 내보내기</h2>
          <div className="flex items-center gap-3">
            <button
              onClick={handleExportPdf}
              disabled={isDownloading}
              className="flex items-center gap-2 px-6 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors disabled:opacity-50"
            >
              {isDownloading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Download className="w-5 h-5" />}
              <span>{isDownloading ? "생성 중..." : "PDF 다운로드"}</span>
            </button>
          </div>
        </div>
      </div>

      {/* =========================================================
          2. PDF 전용 숨김 UI (화면 밖에서 꽉 찬 레이아웃으로 렌더링)
          ========================================================= */}
      <div style={{ position: 'fixed', top: 0, left: '-9999px', zIndex: -1000 }}>
        <div ref={pdfRef} className="w-[820px] bg-white p-12 flex flex-col gap-10">
          <div className="border-b-4 border-blue-600 pb-6">
            <h1 className="text-3xl font-black text-gray-900 tracking-tight">운영 분석 리포트</h1>
            <div className="flex justify-between items-center mt-3">
              <p className="text-gray-500 font-medium">검사 데이터 분석 결과 보고서</p>
              <p className="text-gray-400 text-sm">{new Date().toLocaleString()}</p>
            </div>
          </div>

          <section>
            <div className="grid grid-cols-5 gap-4">
              {[
                { label: "총 검사", val: reportMetrics.total, bg: "bg-slate-50", text: "text-slate-700" },
                { label: "불량(NG)", val: reportMetrics.ngCount, bg: "bg-red-50", text: "text-red-700", sub: `${reportMetrics.ngRate}%` },
                { label: "재검토", val: reportMetrics.reviewCount, bg: "bg-amber-50", text: "text-amber-700" },
                { label: "정상(OK)", val: reportMetrics.okCount, bg: "bg-emerald-50", text: "text-emerald-700" },
                { label: "평균 점수", val: reportMetrics.avgScore, bg: "bg-blue-50", text: "text-blue-700" },
              ].map((item) => (
                <div key={item.label} className={`${item.bg} rounded-2xl p-6 border border-opacity-50 flex flex-col items-center justify-center shadow-sm`}>
                  <p className="text-xs font-bold text-gray-500 mb-2">{item.label}</p>
                  <p className={`text-3xl font-black ${item.text}`}>{item.val}</p>
                  {item.sub && <p className="text-xs font-bold text-red-400 mt-1">{item.sub}</p>}
                </div>
              ))}
            </div>
          </section>

          <div className="grid grid-cols-2 gap-10">
            <div className="bg-white rounded-2xl p-6 border border-gray-100 shadow-sm">
              <h3 className="text-sm font-bold text-gray-800 mb-6 flex items-center gap-2">
                <div className="w-1 h-4 bg-red-500 rounded-full"></div> 불량률 변화 추이
              </h3>
              <div className="h-[280px]">
                <ResponsiveContainer width="100%" height="100%">
                  {/* PDF 차트는 애니메이션을 꺼서 즉시 캡처 가능하도록 설정 */}
                  <LineChart data={defectRateTrend}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                    <XAxis dataKey="time" tick={{fontSize: 11}} axisLine={false} />
                    <YAxis tick={{fontSize: 11}} axisLine={false} />
                    <Line isAnimationActive={false} type="monotone" dataKey="불량률" stroke="#ef4444" strokeWidth={4} dot={{r: 5, fill: '#ef4444'}} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="bg-white rounded-2xl p-6 border border-gray-100 shadow-sm">
              <h3 className="text-sm font-bold text-gray-800 mb-6 flex items-center gap-2">
                <div className="w-1 h-4 bg-orange-500 rounded-full"></div> 결함 유형 분포
              </h3>
              <div className="h-[280px]">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      isAnimationActive={false}
                      data={defectTypeData}
                      cx="50%" cy="50%"
                      innerRadius={70} outerRadius={95}
                      dataKey="value"
                      label={({name, percent}) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {defectTypeData.map((_, index) => <Cell key={index} fill={COLORS[index % COLORS.length]} />)}
                    </Pie>
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-2xl p-8 border border-gray-100 shadow-sm flex-grow">
            <h3 className="text-sm font-bold text-gray-800 mb-8 flex items-center gap-2">
              <div className="w-1 h-4 bg-blue-500 rounded-full"></div> 제품별 주요 불량 건수 (Top 10)
            </h3>
            <div className="h-[350px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={productDefectsFixed10} margin={{ bottom: 50 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                  <XAxis dataKey="name" tick={{fontSize: 11}} interval={0} angle={-30} textAnchor="end" />
                  <YAxis tick={{fontSize: 11}} />
                  <Bar isAnimationActive={false} dataKey="count" fill="#3b82f6" radius={[6, 6, 0, 0]} barSize={45} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}