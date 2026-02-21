// src/app/pages/ReportBuilderPage.tsx
import React, { useMemo } from "react";
import { Download } from "lucide-react";
import type { AnomalyCase } from "../data/mockData";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import {
  buildAggregates,
  toDefectRateTrend,
  toDefectTypePie,
  toProductDefectsFixed10,
} from "../selectors/caseSelectors";

interface ReportBuilderPageProps {
  cases: AnomalyCase[];
}

export function ReportBuilderPage({ cases }: ReportBuilderPageProps) {
  const agg = useMemo(() => buildAggregates(cases), [cases]);

  const reportMetrics = useMemo(() => {
    const total = agg.total;
    const ngCount = agg.ng;
    const reviewCount = agg.review;
    const okCount = agg.ok;

    const ngRate = total ? ((ngCount / total) * 100).toFixed(2) : "0.00";
    const reviewRate = total ? ((reviewCount / total) * 100).toFixed(2) : "0.00";
    const passRate = total ? ((okCount / total) * 100).toFixed(2) : "0.00";

    const avgScore = total ? agg.avgScore.toFixed(3) : "0.000";
    const avgInferenceTime = total ? agg.avgInference.toFixed(1) : "0.0";

    return {
      total,
      ngCount,
      reviewCount,
      okCount,
      ngRate,
      reviewRate,
      passRate,
      avgScore,
      avgInferenceTime,
      representativeCases: agg.topNg,
    };
  }, [agg]);

  const defectRateTrend = useMemo(() => toDefectRateTrend(agg), [agg]);
  const defectTypeData = useMemo(() => toDefectTypePie(agg), [agg]);
  const productDefectsFixed10 = useMemo(() => toProductDefectsFixed10(agg), [agg]);

  const COLORS = ["#ef4444", "#f97316", "#eab308", "#84cc16", "#06b6d4", "#8b5cf6"];

  const handleExport = (format: string) => {
    alert(`리포트를 ${format} 형식으로 내보냅니다.`);
  };

  return (
    <div className="p-8">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 mb-2">운영 리포트</h1>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-6 mb-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">요약 (Executive Summary)</h2>
        <div className="grid grid-cols-5 gap-6 mb-6">
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
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={defectRateTrend}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="time" tick={{ fontSize: 12 }} stroke="#6b7280" />
              <YAxis tick={{ fontSize: 12 }} stroke="#6b7280" />
              <Tooltip />
              <Line type="monotone" dataKey="불량률" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4">결함 타입 분포</h3>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie
                data={defectTypeData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={86}
                dataKey="value"
              >
                {defectTypeData.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-6 mb-6">
        <h3 className="text-lg font-bold text-gray-900 mb-4">제품별 불량 비교</h3>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={productDefectsFixed10} margin={{ bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="name"
              tick={{ fontSize: 12 }}
              stroke="#6b7280"
              interval={0}
              angle={-15}
              textAnchor="end"
              height={60}
            />
            <YAxis tick={{ fontSize: 12 }} stroke="#6b7280" allowDecimals={false} />
            <Tooltip />
            <Legend />
            <Bar dataKey="count" name="불량 건수(NG)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">리포트 내보내기</h2>
        <div className="flex items-center gap-3">
          <button
            onClick={() => handleExport("PDF")}
            className="flex items-center gap-2 px-6 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
          >
            <Download className="w-5 h-5" />
            <span>PDF 다운로드</span>
          </button>
        </div>
        <p className="text-sm text-gray-500 mt-4">리포트에는 위의 모든 차트, 통계가 포함됩니다.</p>
      </div>
    </div>
  );
}
