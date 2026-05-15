import { useState } from "react";
import AppShell, { type Tab } from "./components/layout/AppShell";
import DiagnosisPage from "./components/diagnosis/DiagnosisPage";
import TrainingPage from "./components/training/TrainingPage";
import HistoryPage from "./components/history/HistoryPage";

export default function App() {
  const [tab, setTab] = useState<Tab>("diagnosis");

  return (
    <AppShell activeTab={tab} onTabChange={setTab}>
      {tab === "diagnosis" && <DiagnosisPage />}
      {tab === "training" && <TrainingPage />}
      {tab === "history" && <HistoryPage />}
    </AppShell>
  );
}
