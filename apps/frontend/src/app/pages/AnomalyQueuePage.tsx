// src/app/pages/AnomalyQueuePage.tsx
import React from "react";
import { AnomalyCase } from "../data/mockData";
import { Search } from "lucide-react";

interface AnomalyQueuePageProps {
  cases: AnomalyCase[];
  onCaseClick: (caseId: string) => void;
}

export function AnomalyQueuePage({ cases, onCaseClick }: AnomalyQueuePageProps) {
  return (
    <div className="p-6 space-y-6 bg-gray-50 min-h-full">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">이상 큐</h1>
        <p className="text-gray-500 mt-1">총 {cases.length}개 케이스</p>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <table className="w-full text-sm text-left">
          <thead className="bg-gray-50 text-gray-500 font-medium border-b border-gray-200">
            <tr>
              <th className="px-6 py-3">시간</th>
              <th className="px-6 py-3">케이스 ID</th>
              <th className="px-6 py-3">라인</th>
              <th className="px-6 py-3">교대</th>
              <th className="px-6 py-3">제품</th>
              <th className="px-6 py-3">결함 타입</th>
              <th className="px-6 py-3">Score</th>
              <th className="px-6 py-3">위치</th>
              <th className="px-6 py-3">판정</th>
              <th className="px-6 py-3">심각도</th>
              <th className="px-6 py-3"></th>
            </tr>
          </thead>

          <tbody className="divide-y divide-gray-100">
            {cases.map((c) => (
              <tr key={c.id} className="hover:bg-gray-50 transition-colors">
                <td className="px-6 py-4 text-gray-500 whitespace-nowrap">
                  {c.timestamp.toLocaleDateString()} <br />
                  {c.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </td>

                <td
                  className="px-6 py-4 font-medium text-blue-600 cursor-pointer hover:underline"
                  onClick={() => onCaseClick(c.id)}
                >
                  {c.id}
                </td>

                <td className="px-6 py-4 text-gray-700">{c.line_id}</td>
                <td className="px-6 py-4 text-gray-500">{c.shift}</td>

                <td className="px-6 py-4">
                  <div className="flex flex-col">
                    <span className="text-sm text-gray-700">{c.product_group}</span>
                  </div>
                </td>

                <td className="px-6 py-4 text-gray-700">{c.defect_type === "none" ? "-" : c.defect_type}</td>

                <td className="px-6 py-4">
                  <div className="flex items-center gap-2">
                    <div className="w-16 h-2 bg-gray-100 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${
                          c.anomaly_score > 0.8 ? "bg-red-500" : c.anomaly_score > 0.6 ? "bg-orange-500" : "bg-green-500"
                        }`}
                        style={{ width: `${c.anomaly_score * 100}%` }}
                      />
                    </div>
                    <span className="text-xs font-medium">{c.anomaly_score.toFixed(2)}</span>
                  </div>
                </td>

                <td className="px-6 py-4 text-gray-500">{c.location === "none" ? "-" : c.location}</td>

                <td className="px-6 py-4">
                  <span
                    className={`px-2 py-1 rounded-full text-xs font-medium ${
                      c.decision === "NG"
                        ? "bg-red-100 text-red-700"
                        : c.decision === "REVIEW"
                          ? "bg-orange-100 text-orange-700"
                          : "bg-green-100 text-green-700"
                    }`}
                  >
                    {c.decision}
                  </span>
                </td>

              <td className="px-6 py-4 whitespace-nowrap">
                <span
                  className={`inline-flex items-center justify-center px-2 py-1 rounded border text-xs whitespace-nowrap ${
                    c.severity === "high"
                      ? "bg-red-50 border-red-200 text-red-700"
                      : c.severity === "med"
                        ? "bg-orange-50 border-orange-200 text-orange-700"
                        : "bg-gray-50 border-gray-200 text-gray-600"
                  }`}
  >
                  {c.severity === "high" ? "높음" : c.severity === "med" ? "중간" : "낮음"}
                </span>
              </td>
              
                <td className="px-6 py-4 text-right">
                  <button
                    onClick={() => onCaseClick(c.id)}
                    className="p-1 hover:bg-gray-100 rounded text-gray-400 hover:text-blue-600"
                  >
                    <Search className="w-4 h-4" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
