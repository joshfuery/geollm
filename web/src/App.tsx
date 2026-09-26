import { useState } from "react";
import { BrowseView } from "./BrowseView";
import { LiveView } from "./LiveView";

type Mode = "live" | "browse";

export default function App() {
  const [mode, setMode] = useState<Mode>("live");
  const header = (
    <header className="masthead">
      <div className="wordmark">GeoAgent<span>Bench</span></div>
      <nav className="tabs" role="tablist">
        <button role="tab" aria-selected={mode === "live"} className={mode === "live" ? "on" : ""} onClick={() => setMode("live")}>
          Live
        </button>
        <button role="tab" aria-selected={mode === "browse"} className={mode === "browse" ? "on" : ""} onClick={() => setMode("browse")}>
          Results
        </button>
      </nav>
    </header>
  );
  return <div className="app">{mode === "live" ? <LiveView header={header} /> : <BrowseView header={header} />}</div>;
}
