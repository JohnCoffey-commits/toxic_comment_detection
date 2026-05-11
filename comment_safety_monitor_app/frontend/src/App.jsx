import { useState } from "react";
import Layout from "./components/Layout.jsx";
import TextCheck from "./pages/TextCheck.jsx";
import FileReview from "./pages/FileReview.jsx";

export default function App() {
  const [page, setPage] = useState("text");

  return (
    <Layout page={page} onNavigate={setPage}>
      {page === "text" ? <TextCheck /> : <FileReview />}
    </Layout>
  );
}
