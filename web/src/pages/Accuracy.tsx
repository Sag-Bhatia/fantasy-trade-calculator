import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "../api";
import { Card, ErrorNote, Spinner } from "../components/ui";
import { useLoad } from "../hooks";

export function Accuracy() {
  const report = useLoad("report", () => api.report());
  return (
    <main className="mx-auto flex max-w-[1100px] flex-col gap-6 px-5 py-7 md:px-10">
      <h1 className="font-display text-4xl leading-[0.95] font-black tracking-tight md:text-[56px]">How accurate is it?</h1>
      <p className="max-w-[64ch] text-lg leading-relaxed">
        Every number below comes from replaying real past NFL seasons through the same engine that grades your trades. Nothing is typed in by hand.
      </p>
      {report.loading && <Spinner label="Loading the report" />}
      {report.error && <ErrorNote>{report.error}</ErrorNote>}
      {report.data && (
        <Card className="report p-6 md:p-9">
          <Markdown remarkPlugins={[remarkGfm]}>{report.data.markdown.replace(/^# .*\n/, "")}</Markdown>
        </Card>
      )}
    </main>
  );
}
