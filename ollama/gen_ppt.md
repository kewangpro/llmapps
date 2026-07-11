# Product Requirements Document: AI PDF-to-PowerPoint Converter

## 1.0 Executive Summary

This document outlines the product requirements for the **AI PDF-to-PowerPoint Converter**, a desktop application designed to automate the conversion of PDF documents into fully editable PowerPoint presentations. The application leverages the power of local Large Language Models (LLMs) through the Ollama framework to analyze, summarize, and structure content into a slide-based format. The primary goal is to provide a privacy-focused, efficient tool for professionals, educators, and students, drastically reducing the manual effort required to create presentations from existing documents. The application will feature both a user-friendly graphical interface (GUI) and a command-line interface (CLI) for flexible workflow integration.

## 2.0 Problem Statement

Creating presentations from dense PDF documents (such as reports, research papers, or articles) is a tedious and time-consuming process. Users must manually read through the content, identify key points, summarize information, and then copy-paste the results into a presentation application, all while trying to maintain logical structure and consistent formatting. This manual process is not only inefficient but also prone to errors and inconsistencies. Furthermore, for users dealing with sensitive or proprietary information, using cloud-based conversion tools poses a significant data privacy risk. There is a clear need for an intelligent tool that can automate this conversion process locally, ensuring both efficiency and data security.

## 3.0 Solution Overview

The AI PDF-to-PowerPoint Converter is a Python-based desktop application that provides a seamless bridge between PDF documents and PowerPoint presentations. The core workflow is as follows:

1.  **Input:** The user selects a PDF file via the application's GUI or specifies a path via the CLI.
2.  **Text Extraction:** The application parses the PDF and extracts its raw text content.
3.  **AI Processing:** The extracted text is broken into manageable chunks and sent to a user-selected Ollama LLM running on the user's local machine. A carefully engineered prompt instructs the model to generate a structured slide outline, complete with a title and several bullet points for each slide.
4.  **Presentation Generation:** The application parses the LLM's output and programmatically generates a standard `.pptx` PowerPoint file. Each slide is populated with the title and content as defined by the AI.

This approach ensures that all data processing occurs locally, preserving user privacy. By leveraging the summarization and structuring capabilities of LLMs, the application delivers a high-quality, relevant, and well-organized presentation with minimal user effort.

## 4.0 User Stories

-   **As a business consultant,** I want to quickly convert a 50-page market research PDF into a PowerPoint presentation so I can present the key findings to my client without spending hours on manual summarization and slide creation.
-   **As a university professor,** I want to transform a dense academic paper from a PDF into a slide deck so I can create engaging lecture materials for my students.
-   **As a student,** I want to turn my PDF lecture notes and research articles into a presentation format so I can easily prepare for class presentations and study groups.
-   **As a developer,** I want to use a command-line interface to batch-process multiple PDF documents into presentations as part of an automated content pipeline.

## 5.0 Functional Requirements

### 5.1 File Handling
-   **FR1.1:** The application MUST allow users to select a PDF file from their local file system through a standard file dialog in the GUI.
-   **FR1.2:** The application MUST support PDF file input via a command-line argument (`--pdf_file`).
-   **FR1.3:** The application MUST extract all text content from the provided PDF. It should gracefully handle empty pages or pages without text.
-   **FR1.4:** The application MUST provide a clear error message to the user if a PDF file is corrupt, password-protected, or cannot be read.

### 5.2 AI & Conversion Process
-   **FR2.1:** The application MUST integrate with a locally running Ollama instance.
-   **FR2.2:** The application MUST automatically detect and list available Ollama models for the user to choose from.
-   **FR2.3:** The application SHOULD filter the model list to primarily show chat/instruction-tuned models and exclude embedding-focused models.
-   **FR2.4:** The application MUST send extracted text to the selected LLM with a prompt engineered to produce slide titles and 3-5 bullet points per slide.
-   **FR2.5:** The application MUST handle large documents by splitting the text into smaller chunks (e.g., ~3000 characters) to fit within the LLM's context window, processing each chunk sequentially.

### 5.3 Presentation Output
-   **FR3.1:** The application MUST generate a standard, editable PowerPoint file (`.pptx`).
-   **FR3.2:** The output file MUST be named based on the original PDF file (e.g., `original_name_presentation.pptx`).
-   **FR3.3:** The application MUST allow users to select from a predefined list of standard PowerPoint slide layouts (e.g., "Title and Content," "Title Only").
-   **FR3.4:** The application MUST parse the LLM output and map it correctly to titles and content placeholders in the PowerPoint slides.

### 5.4 User Interface (GUI)
-   **FR4.1:** The GUI MUST provide a clear, intuitive interface for all user actions.
-   **FR4.2:** The GUI MUST feature a dropdown menu for Ollama model selection, including a "Refresh" button to update the list.
-   **FR4.3:** The GUI MUST have a dropdown menu to select the desired slide layout.
-   **FR4.4:** The GUI MUST display real-time progress updates and logs in a dedicated text area.
-   **FR4.5:** The GUI MUST show a progress bar during the conversion process to indicate that the application is working.
-   **FR4.6:** The GUI MUST provide a "Cancel" button to stop the conversion process mid-operation.
-   **FR4.7:** All processing MUST be done in a background thread to keep the GUI responsive.

## 6.0 Technical Requirements

-   **TR1.1:** **Platform:** The application must be cross-platform (Windows, macOS, Linux).
-   **TR1.2:** **Backend:** Python 3.7+.
-   **TR1.3:** **GUI Framework:** PyQt6.
-   **TR1.4:** **Core Libraries:**
    -   `python-pptx` for PowerPoint generation.
    -   `PyPDF2` for PDF text extraction.
    -   `aiohttp` for asynchronous communication with the Ollama API.
-   **TR1.5:** **Dependencies:** The application requires a local installation of Ollama, accessible at `http://127.0.0.1:11434`.

## 7.0 UI Requirements

-   **UIR1.1:** The main window shall be titled "PDF to PPTX with Ollama LLM".
-   **UIR1.2:** The interface shall contain the following components, organized vertically:
    1.  An "Ollama Model" selection area with a dropdown menu and a "Refresh Models" button.
    2.  A "Slide Layout" selection dropdown.
    3.  A "Select PDF" button to initiate the process.
    4.  A "Cancel" button, enabled only during processing.
    5.  A progress bar, visible only during processing.
    6.  A read-only text area for displaying status messages and logs.
-   **UIR1.3:** The model dropdown should display the model name and its size (e.g., "gemma3 (8.9GB)").
-   **UIR1.4:** The application state should be clear at all times (e.g., buttons should be enabled/disabled appropriately).

## 8.0 Success Metrics

-   **SM1:** **Conversion Success Rate:** >95% of conversion attempts for valid, text-based PDFs complete without error.
-   **SM2:** **User Satisfaction:** A high rating of output quality, measured by user feedback (if a feedback mechanism is implemented).
-   **SM3:** **Performance:** Average processing time for a 20-page PDF is under 2 minutes on a standard machine with a 7B parameter model.
-   **SM4:** **Adoption:** A steady increase in the number of downloads and active users (if distributed).

## 9.0 Risk Assessment

| Risk ID | Risk Description | Probability | Impact | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **R1** | **Low-Quality LLM Output:** The LLM generates irrelevant, inaccurate, or poorly formatted slide content (hallucination). | Medium | High | - Engineer and refine the system prompt for better consistency. <br> - Allow users to easily edit the generated presentation. <br> - Log LLM output for debugging to identify and fix prompt issues. |
| **R2** | **PDF Parsing Failures:** The application fails to extract text from image-based PDFs, complex layouts, or corrupted files. | High | Medium | - Use a robust PDF parsing library. <br> - Clearly communicate to the user that the tool works best with text-based PDFs. <br> - Implement error handling to inform the user when text cannot be extracted. |
| **R3** | **Performance Bottlenecks:** The conversion process is excessively slow for large PDFs or when using large LLM models. | Medium | Medium | - Process text in asynchronous chunks. <br> - Provide clear, real-time feedback (progress bar, logs) so the user knows the app is working. <br> - Advise users on the performance trade-offs of different model sizes. |
| **R4** | **Dependency Issues:** The user does not have Ollama installed or running correctly. | High | High | - Provide clear, prominent instructions on how to install and run Ollama. <br> - Implement a connection check on startup with a user-friendly error message if the Ollama API is not reachable. |
| **R5** | **Poor User Experience:** The generated presentation has poor formatting or nonsensical structure, requiring heavy manual edits. | Medium | High | - Continuously improve the prompt and the logic that parses the LLM response. <br> - Default to simple, clean slide layouts that are less prone to formatting errors. <br> - Gather user feedback on output quality to guide improvements. |

## 10.0 Implementation Timeline

### Phase 1: Core Functionality (Weeks 1-4)
- Complete PDF text extraction capabilities
- Implement basic Ollama API integration
- Develop core slide generation logic
- Create minimal viable CLI interface

### Phase 2: GUI Development (Weeks 5-8)
- Build PyQt6 user interface
- Implement model selection and refresh functionality
- Add progress tracking and cancellation features
- Integrate background processing threads

### Phase 3: Enhancement & Polish (Weeks 9-12)
- Improve error handling and user feedback
- Optimize performance for large documents
- Add comprehensive logging and debugging features
- Conduct user testing and iterate on UX

### Phase 4: Testing & Release (Weeks 13-16)
- Comprehensive testing across platforms
- Documentation creation
- Package distribution preparation
- Performance optimization and bug fixes

## 11.0 Future Enhancements

- **Batch Processing:** Support for processing multiple PDFs in a single operation
- **Template Support:** Custom PowerPoint templates and themes
- **OCR Integration:** Support for image-based PDFs through optical character recognition
- **Cloud Model Support:** Integration with cloud-based LLM services as an option
- **Advanced Formatting:** More sophisticated slide layouts and design options
- **Export Options:** Additional output formats (Google Slides, HTML presentations)