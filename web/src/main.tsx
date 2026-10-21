import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { Shell } from "./components/Shell";
import { Builder } from "./pages/Builder";
import { Finder } from "./pages/Finder";
import { LeaguePage } from "./pages/League";
import { Result } from "./pages/Result";
import { Welcome } from "./pages/Welcome";
import "./styles.css";

// the Markdown renderer is only needed on the accuracy page
const Accuracy = lazy(() => import("./pages/Accuracy").then((m) => ({ default: m.Accuracy })));

const router = createBrowserRouter([
  { path: "/", element: <Welcome /> },
  {
    path: "/:league/:team",
    element: <Shell />,
    children: [
      { index: true, element: <Builder /> },
      { path: "result", element: <Result /> },
      { path: "find", element: <Finder /> },
      { path: "league", element: <LeaguePage /> },
      { path: "accuracy", element: <Suspense fallback={null}><Accuracy /></Suspense> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
