---
name: nextcloud-publish-asset
skill_type: skill
description: >-
  Publish a produced business asset (a photo, video, marketing graphic or document from the
  creative suite) to the team folder in Nextcloud and share it: copy it from the production
  archive into Team/<campaign>, grant the team group access once, and create an expiring public
  link for external use. Use when an asset is ready for the team or for a customer, partner or
  social post. Do NOT use to produce the asset (the creative suite skills), for machine delivery
  or a website/CDN URL (the object storage skills), or to post to social channels.
license: MIT
tags: [nextcloud, publishing, assets, marketing, sharing, mcp]
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Publish an asset to the team folder and share it

## Where assets live (routing)
Every creative-suite run writes its outputs to two stores at once:

| Store | Location | Use it for |
|-------|----------|------------|
| Object storage (canonical) | bucket `artcraft`, key `<app>/<date>/<run>/<name>`, with a signed URL | durable archive, machine delivery, website/CDN, social scheduling |
| Nextcloud (collaboration) | the production account's `ArtCraft/<app>/<date>/<run>/<name>`, shared read-only into the operator account as `ArtCraft/` | people: review, comment, versions, team folders, public links |

So the asset is normally already in Nextcloud under `ArtCraft/`. This skill moves it into the
business's working space and shares it; it never edits the production archive. When an asset
exists only in object storage, fetch its bytes with the object storage skill and upload it with
`nextcloud_write_file(path, content=<base64>, encoding="base64")` (keep this to files up to tens
of MB; bigger files should be produced again so the suite writes them to both stores).

## Inputs
- `asset`: the path the suite reported (`ArtCraft/<app>/<date>/<run>/<name>`) or a name to search.
- `campaign`: the team folder name under `Team/` (for example `2026-11-launch`).
- `team_group`: the group that works on the campaign (`nextcloud_list_groups`); ask if unknown.
- `public`: whether an external link is wanted, and its expiry (default 14 days).

## Contract (tool-use order)
1. Locate: `nextcloud_list_files(path="ArtCraft/<app>/<date>/<run>")`, or
   `nextcloud_search_files(query=<name>, path="ArtCraft", mime_prefix="image/")` (`video/` for video).
2. `nextcloud_create_folder(path="Team/<campaign>")` (and `Team` the first time); an "already exists"
   error is fine.
3. `nextcloud_copy_item(source_path=<asset>, dest_path="Team/<campaign>/<name>")`. Copy, never move:
   the archive stays complete. If the name is taken, add the run id to the name; never overwrite.
4. `nextcloud_list_shares(path="Team/<campaign>")`: if no `share_type` 1 share for `team_group`, run
   `nextcloud_create_share(path="Team/<campaign>", share_type="group", share_with=<team_group>, permissions=1)`
   (use 15 when the team should also edit and upload).
5. Only when `public` is wanted: `nextcloud_create_share(path="Team/<campaign>/<name>", share_type="public_link", expire_date=<YYYY-MM-DD>, label=<campaign>)`;
   add `password` for anything not meant for open distribution. Read `url` from the result.
6. Verify: `nextcloud_get_share(share_id)` (type, permissions, expiry) and `nextcloud_get_properties`
   on the copy (size and type match the original).
7. Optional, internal only: `nextcloud_send_message(token, "New asset for <campaign>: <path or url>")`
   in the team's private Talk conversation.

## Report
Return the team-folder path, the share ids, the public URL and its expiry, and the original
archive path (plus the object-storage key when known) so the asset stays traceable.

## Guardrails
- Never create a public link without an explicit request; always set an expiry.
- Never email recipients from Nextcloud (`send_mail` stays false) and never post to public channels.
- Do not delete or modify anything under `ArtCraft/`.
- To withdraw an asset: `nextcloud_delete_share(share_id)`; the file stays in the team folder.
