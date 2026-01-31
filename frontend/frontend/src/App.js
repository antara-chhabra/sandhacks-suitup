import React from "react";
import ResumeUpload from "./components/ResumeUpload";

function App() {
  const handleResumeUpload = (file) => {
    console.log("Resume file ready:", file);
  };

  return (
    <div className="App">
      <h1>Sameeksha - Resume Upload</h1>
      <ResumeUpload onUpload={handleResumeUpload} />
    </div>
  );
}

export default App;
