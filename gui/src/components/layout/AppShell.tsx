import type { ReactNode } from "react";
import Sidebar from "./Sidebar";
import Header from "./Header";

export type Tab = "diagnosis" | "training" | "history" | "review";

interface Props {
  activeTab: Tab;
  onTabChange: (tab: Tab) => void;
  children: ReactNode;
}

export default function AppShell({ activeTab, onTabChange, children }: Props) {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar activeTab={activeTab} onTabChange={onTabChange} />
      <div className="flex flex-1 flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-y-auto p-6">{children}</main>
      </div>
    </div>
  );
}
