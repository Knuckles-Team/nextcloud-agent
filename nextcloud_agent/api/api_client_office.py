"""Office documents and collaborative editing (CONN-NEXTCLOUD-AGENT-R014)."""

from nextcloud_agent.api.api_client_base import BaseApiClient

_TEMPLATES = "apps/files/api/v1/templates"
_TEXT_TYPES = (".md", ".markdown", ".txt")


class NextcloudOffice(BaseApiClient):
    """Office documents (Nextcloud Office on the CODE server) and collaborative
    markdown (Text): create a document from a template and open an editing session."""

    def list_document_templates(self) -> list[dict]:
        """List the document types the server can create (document, spreadsheet,
        presentation, diagram, markdown) with their extensions and templates.

        GET /ocs/v2.php/apps/files/api/v1/templates
        """
        return self.ocs_request("GET", _TEMPLATES) or []

    def create_document(self, path: str, template_path: str | None = None) -> dict:
        """Create one new document at ``path``; the extension picks the type
        (``.docx``, ``.xlsx``, ``.pptx``, ``.odg``, ``.md``).

        POST /ocs/v2.php/apps/files/api/v1/templates/create

        ``template_path`` is a template ``filename`` from
        list_document_templates; omitted, the document starts empty. Everyone
        with access to the file edits it together in real time.
        """
        self._get_full_url(path)
        data = {"filePath": "/" + path.strip("/"), "templateType": "user"}
        if template_path:
            data["templatePath"] = template_path
        return self.ocs_request("POST", f"{_TEMPLATES}/create", data=data)

    def open_document(self, path: str) -> dict:
        """Open one document for editing and return a one-time editor ``url``
        for the connected account (Office for office files, Text for markdown).

        POST /ocs/v2.php/apps/richdocuments/api/v1/document
        """
        self._get_full_url(path)
        if path.lower().endswith(_TEXT_TYPES):
            return self.ocs_request(
                "POST",
                "apps/files/api/v1/directEditing/open",
                data={"path": "/" + path.strip("/"), "editorId": "text"},
            )
        file_id = self.get_properties(path).get("file_id")
        if not file_id:
            raise FileNotFoundError("Configured remote path was not found")
        return self.ocs_request(
            "POST", "apps/richdocuments/api/v1/document", data={"fileId": int(file_id)}
        )
