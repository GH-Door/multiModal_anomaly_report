// src/app/App.tsx
import React, { useMemo, useState } from "react";
import { Sidebar } from "./components/Sidebar";
import { FilterBar, FilterState } from "./components/FilterBar";
import { LoadingState } from "./components/LoadingState";
import { EmptyState } from "./components/EmptyState";

import { OverviewPage } from "./pages/OverviewPage";
import { AnomalyQueuePage } from "./pages/AnomalyQueuePage";
import { CaseDetailPage } from "./pages/CaseDetailPage";
import { ReportBuilderPage } from "./pages/ReportBuilderPage";
import { SettingsPage } from "./pages/SettingsPage";

import { mockCases, AnomalyCase } from "./data/mockData";
import { mockAlerts, Alert, NotificationSettings } from "./data/AlertData";
import { getDateRangeWindow } from "./utils/dateUtils";
import { clamp01 } from "./utils/number";

import { useReportCases } from "./hooks/useReportCases";
import { useLocalStorageState } from "./hooks/useLocalStorageState";

const MODEL_VERSION: Record<string, string> = {
  PatchCore: "v2.3.1",
  EfficientAD: "v3.1.0",
};

const DEFAULT_NOTI: NotificationSettings = {
  highSeverity: true,
  reviewRequest: true,
  dailyReport: false,
  systemError: true,
  consecutiveDefects: true,
};

// Static options for useReportCases to avoid re-renders
const REPORT_CASES_OPTIONS = {
  query: {},
  pageSize: 500,
  maxItems: 5000,
};

// Static options for useLocalStorageState
const MODEL_STORAGE_OPTIONS = {
  serialize: (v: string) => v,
  deserialize: (raw: string) => raw || "PatchCore",
};

const THRESHOLD_STORAGE_OPTIONS = {
  serialize: (v: number) => String(v),
  deserialize: (raw: string) => clamp01(Number(raw)),
  normalize: clamp01,
};

const NOTIFICATION_STORAGE_OPTIONS = {
  normalize: (v: NotificationSettings) => ({ ...DEFAULT_NOTI, ...(v ?? {}) }),
};

export default function App() {
  const [currentPage, setCurrentPage] = useState<string>("overview");
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  const [alerts] = useState<Alert[]>(mockAlerts);

  const [activeModel, setActiveModel] = useLocalStorageState<string>(
    "activeModel", 
    "PatchCore", 
    MODEL_STORAGE_OPTIONS
  );

  const [threshold, setThreshold] = useLocalStorageState<number>(
    "threshold", 
    0.65, 
    THRESHOLD_STORAGE_OPTIONS
  );

  const [notificationSettings, setNotificationSettings] =
    useLocalStorageState<NotificationSettings>(
      "notificationSettings", 
      DEFAULT_NOTI, 
      NOTIFICATION_STORAGE_OPTIONS
    );

  const { cases: backendCases, loading, error, refetch } = useReportCases(REPORT_CASES_OPTIONS);

  const cases: AnomalyCase[] = error ? mockCases : backendCases;

  const [filters, setFilters] = useState<FilterState>({
    dateRange: "today",
    line: "all",
    productGroup: "all",
    defectType: "all",
    decision: "all",
    scoreRange: [0, 1],
  });

  const casesWithSettings = useMemo(() => {
    const version = MODEL_VERSION[activeModel] ?? "v1.0.0";
    return cases.map((c) => ({
      ...c,
      model_name: activeModel,
      model_version: version,
      threshold,
    }));
  }, [cases, activeModel, threshold]);

  const filteredCases = useMemo(() => {
    const window = getDateRangeWindow(filters.dateRange);

    return casesWithSettings.filter((c) => {
      if (window) {
        const t = c.timestamp.getTime();
        if (t < window.from.getTime() || t > window.to.getTime()) return false;
      }
      if (filters.line !== "all" && c.line_id !== filters.line) return false;
      if (filters.productGroup !== "all" && c.product_group !== filters.productGroup) return false;
      if (filters.defectType !== "all" && c.defect_type !== filters.defectType) return false;
      if (filters.decision !== "all" && c.decision !== filters.decision) return false;
      if (c.anomaly_score < filters.scoreRange[0] || c.anomaly_score > filters.scoreRange[1]) return false;
      return true;
    });
  }, [casesWithSettings, filters]);

  const handleNavigate = (page: string) => {
    setCurrentPage(page);
    setSelectedCaseId(null);
  };

  const handleCaseClick = (caseId: string) => {
    setSelectedCaseId(caseId);
    setCurrentPage("detail");
  };

  const currentCase = selectedCaseId
    ? casesWithSettings.find((c) => c.id === selectedCaseId) ?? null
    : null;

  const renderPage = () => {
    if (loading && !error) {
      return <LoadingState title="불러오는 중" message="백엔드에서 검사 데이터를 가져오고 있습니다." />;
    }

    if (error && currentPage !== "settings") {
      return (
        <div className="p-6">
          <EmptyState
            type="error"
            title="백엔드 데이터 로딩 실패"
            description="현재는 Mock 데이터를 사용 중입니다. 서버 실행/URL/CORS를 확인해 주세요."
          />
          <div className="flex justify-center">
            <button
              onClick={refetch}
              className="px-4 py-2 bg-gray-900 text-white rounded-lg hover:bg-gray-800"
            >
              다시 시도
            </button>
          </div>
        </div>
      );
    }

    if (currentPage === "detail" && currentCase) {
      return (
        <CaseDetailPage
          caseData={currentCase}
          onBackToQueue={() => setCurrentPage("queue")}
          onBackToOverview={() => setCurrentPage("overview")}
        />
      );
    }

    switch (currentPage) {
      case "overview":
        return <OverviewPage cases={filteredCases} alerts={alerts} activeModel={activeModel} />;
      case "queue":
        return <AnomalyQueuePage cases={filteredCases} onCaseClick={handleCaseClick} />;
      case "report":
        return <ReportBuilderPage cases={filteredCases} />;
      case "settings":
        return (
          <SettingsPage
            activeModel={activeModel}
            onModelChange={setActiveModel}
            threshold={threshold}
            onThresholdChange={setThreshold}
            notifications={notificationSettings}
            onNotificationsChange={setNotificationSettings}
          />
        );
      default:
        return <OverviewPage cases={filteredCases} alerts={alerts} activeModel={activeModel} />;
    }
  };

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar currentPage={currentPage} onNavigate={handleNavigate} />

      <div className="flex-1 flex flex-col overflow-hidden">
        {(currentPage === "overview" || currentPage === "queue" || currentPage === "report") && (
          <FilterBar filters={filters} onFilterChange={setFilters} />
        )}
        <div className="flex-1 overflow-y-auto">{renderPage()}</div>
      </div>
    </div>
  );
}
