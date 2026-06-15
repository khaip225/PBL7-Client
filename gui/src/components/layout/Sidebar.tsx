import { Stethoscope, Brain, Clock, ClipboardCheck, Trash2 } from "lucide-react";
import type { Tab } from "./AppShell";

const items: { id: Tab; label: string; icon: typeof Stethoscope }[] = [
  { id: "diagnosis", label: "Diagnosis", icon: Stethoscope },
  { id: "training", label: "FL Training", icon: Brain },
  { id: "review", label: "Review Data", icon: ClipboardCheck },
  { id: "history", label: "History", icon: Clock },
  { id: "trash", label: "Trash", icon: Trash2 },
];

interface Props {
  activeTab: Tab;
  onTabChange: (tab: Tab) => void;
}

export default function Sidebar({ activeTab, onTabChange }: Props) {
  return (
    <aside className="flex w-56 flex-col bg-gray-900 border-r border-gray-800">
      <div className="flex items-center gap-2.5 px-5 py-4 border-b border-gray-800">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-600 text-white text-sm font-bold">
          P7
        </div>
        <span className="text-sm font-semibold tracking-wide">PBL7 Client</span>
      </div>
      <nav className="flex flex-col gap-1 p-3">
        {items.map(({ id, label, icon: Icon }) => {
          const active = activeTab === id;
          return (
            <button
              key={id}
              onClick={() => onTabChange(id)}
              className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                active
                  ? "bg-blue-600/15 text-blue-400"
                  : "text-gray-400 hover:bg-gray-800 hover:text-gray-200"
              }`}
            >
              <Icon size={18} />
              {label}
            </button>
          );
        })}
      </nav>
    </aside>
  );
}
