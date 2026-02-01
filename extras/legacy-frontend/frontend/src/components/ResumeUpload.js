import React, { useState } from "react";

function ResumeUpload() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");

  const handleUpload = async () => {
    if (!file) {
      setStatus("Please select a resume");
      return;
    }

    const formData = new FormData();
    formData.append("resume", file);

    try {
      setStatus("Uploading...");
      const response = await fetch("http://localhost:5000/upload-resume", {
        method: "POST",
        body: formData
      });

      const data = await response.json();
      setStatus(data.message);
    } catch (err) {
      console.error(err);
      setStatus("Upload failed");
    }
  };

  return (
    <div>
      <h2>Resume Upload</h2>
      <input
        type="file"
        accept=".pdf,.doc,.docx"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <br /><br />
      <button onClick={handleUpload}>Upload Resume</button>
      <p>{status}</p>
    </div>
  );
}

export default ResumeUpload;
