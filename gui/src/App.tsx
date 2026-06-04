import { useState } from "react";
import AppShell, { type Tab } from "./components/layout/AppShell";
import DiagnosisPage from "./components/diagnosis/DiagnosisPage";
import TrainingPage from "./components/training/TrainingPage";
import HistoryPage from "./components/history/HistoryPage";
import ReviewPage from "./components/review/ReviewPage";
import TrashPage from "./components/review/TrashPage";

export default function App() {
  const [tab, setTab] = useState<Tab>("diagnosis");

  return (
    <AppShell activeTab={tab} onTabChange={setTab}>
      {tab === "diagnosis" && <DiagnosisPage />}
      {tab === "training" && <TrainingPage />}
      {tab === "review" && <ReviewPage />}
      {tab === "history" && <HistoryPage />}
      {tab === "trash" && <TrashPage />}
    </AppShell>
  );
}
