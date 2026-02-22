// src/app/App.tsx
import React, { useMemo, useState, useEffect } from "react";
import { AlertOctagon } from "lucide-react"; // 팝업용 아이콘 추가

import { Sidebar } from "./components/Sidebar";
import { FilterBar, FilterState } from "./components/FilterBar";
import { LoadingState } from "./components/LoadingState";
import { EmptyState } from "./components/EmptyState";

import { OverviewPage } from "./pages/OverviewPage";
import { AnomalyQueuePage } from "./pages/AnomalyQueuePage";
import { CaseDetailPage } from "./pages/CaseDetailPage";
import { ReportBuilderPage } from "./pages/ReportBuilderPage";
import { SettingsPage } from "./pages/SettingsPage";

import { AnomalyCase } from "./data/mockData";
import { Alert, NotificationSettings } from "./data/AlertData";
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

const REPORT_CASES_OPTIONS = {
  query: {},
  pageSize: 500,
  maxItems: 5000,
};

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
  
  // 팝업 알림 관련 상태
  const [showToast, setShowToast] = useState(false);
  const [isExiting, setIsExiting] = useState(false);
  const [lastNgCount, setLastNgCount] = useState(0);

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

  // 백엔드 데이터 연동 훅
  const { cases: backendCases, loading, error, refetch } = useReportCases(REPORT_CASES_OPTIONS);

  // 1. 실시간 폴링 (5초마다 백엔드 데이터 갱신)
  useEffect(() => {
    const interval = setInterval(() => {
      refetch(); 
    }, 5000);
    return () => clearInterval(interval);
  }, [refetch]);

  const cases: AnomalyCase[] = backendCases;

  // 2. 신규 결함 감지 및 토스트 팝업 트리거
  useEffect(() => {
    const currentNgCount = cases.filter(c => c.decision === "NG").length;
    
    // 이전에 기록된 개수가 있고, 현재 NG 개수가 더 늘어났다면 신규 발생으로 간주
    if (lastNgCount > 0 && currentNgCount > lastNgCount) {
      setShowToast(true);
      setIsExiting(false);

      const timer = setTimeout(() => {
        setIsExiting(true);
        setTimeout(() => setShowToast(false), 400); 
      }, 5000); 

      return () => clearTimeout(timer);
    }
    
    setLastNgCount(currentNgCount);
  }, [cases, lastNgCount]);

  // 3. 실시간 알림창 리스트 (우측 사이드바용)
  const alerts: Alert[] = useMemo(() => {
    return cases
      .filter(c => c.decision === "NG")
      .map(c => ({
        id: c.id,
        type: "critical",
        severity: "high",
        title: "결함 감지",
        description: `${c.product_group} 제품에서 결함 발생`,
        timestamp: c.timestamp,
        line_id: c.line_id,
        defect_type: c.defect_type
      }))
      .slice(0, 5); // 최신 5개만 유지
  }, [cases]);

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
    if (loading && !error && cases.length === 0) {
      return <LoadingState title="데이터 로드 중" message="백엔드 서버에서 검사 이력을 가져오고 있습니다." />;
    }

    if (error && currentPage !== "settings") {
      return (
        <div className="p-6">
          <EmptyState
            type="error"
            title="연동 실패"
            description="백엔드 서버와 통신할 수 없습니다. 서버 실행 상태를 확인하세요."
          />
          <div className="flex justify-center mt-4">
            <button onClick={refetch} className="px-4 py-2 bg-gray-900 text-white rounded-lg">다시 시도</button>
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
    <div className="flex h-screen bg-gray-50 relative">
      
      {/* 팝업(Toast) 알림 UI */}
      {showToast && (
        <div className={`fixed top-6 right-6 z-[9999] ${isExiting ? 'animate-toast-out' : 'animate-toast-in'}`}>
          <div className="bg-red-600 text-white px-6 py-4 rounded-lg shadow-2xl flex items-center space-x-3 border-2 border-red-400">
            <div className="bg-white p-1 rounded-full">
              <AlertOctagon className="w-6 h-6 text-red-600" />
            </div>
            <div>
              <p className="font-bold">신규 결함 감지!</p>
              <p className="text-sm opacity-90 font-medium">최신 검사에서 불량이 발견되었습니다.</p>
            </div>
            <button onClick={() => {
              setIsExiting(true);
              setTimeout(() => setShowToast(false), 400);
            }}>✕</button>
          </div>
        </div>
      )}

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