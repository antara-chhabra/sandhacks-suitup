import React, { useState } from "react";

const ResumeUpload = ({ onUpload }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleUpload = () => {
    if (!selectedFile) {
      setUploadStatus("Please select a file first.");
      return;
    }
    console.log("File ready for upload:", selectedFile);
    setUploadStatus("File ready for upload!");
    onUpload && onUpload(selectedFile);
  };

  return (
    <div>
      <h2>Upload Your Resume</h2>
      <input type="file" accept=".pdf,.doc,.docx" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload</button>
      <p>{uploadStatus}</p>
    </div>
  );
};

export default ResumeUpload;
