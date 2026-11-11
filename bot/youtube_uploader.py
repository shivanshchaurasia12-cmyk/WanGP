import os

PRIVACY = os.environ.get("YT_PRIVACY", "private")


def _secret(name):
    try:
        from bot import config
        v = config.get_secret(name)
        if v:
            return v.strip()
    except Exception:
        pass
    return os.environ[name].strip()


def upload_short(path, title, description="", tags=None):
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    creds = Credentials(
        None,
        refresh_token=_secret("YT_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=_secret("YT_CLIENT_ID"),
        client_secret=_secret("YT_CLIENT_SECRET"),
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    yt = build("youtube", "v3", credentials=creds)
    body = {
        "snippet": {
            "title": (title + " #Shorts")[:100],
            "description": (description + "\n\n#Shorts #Kids #Cartoon")[:4900],
            "tags": (tags or [])[:15],
            "categoryId": "24",
        },
        "status": {"privacyStatus": PRIVACY, "selfDeclaredMadeForKids": True},
    }
    req = yt.videos().insert(part="snippet,status", body=body,
                             media_body=MediaFileUpload(path, resumable=True))
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    print("Uploaded video id:", resp["id"])
    return resp["id"]
