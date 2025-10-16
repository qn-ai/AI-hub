import time
import json
from typing import Optional, Dict, Any
import requests
import boto3
from botocore.exceptions import ClientError
import jwt

class CognitoAuth:
    def __init__(self, region: str, user_pool_id: str, app_client_id: str):
        self.region = region
        self.user_pool_id = user_pool_id
        self.app_client_id = app_client_id
        self.idp = boto3.client("cognito-idp", region_name=self.region)
        self._jwks_cache: Optional[Dict[str, Any]] = None
        self._jwks_cache_ts: float = 0.0
        self._jwks_ttl_sec = 60 * 60  # 1 hour

    # ----- Public API -----

    def login_user_password(self, username: str, password: str) -> Dict[str, Any]:
        """
        Perform USER_PASSWORD_AUTH and return tokens on success:
        {
            'IdToken': str,
            'AccessToken': str,
            'RefreshToken': str,
            'ExpiresIn': int,   # seconds
            'TokenType': 'Bearer'
        }
        """
        try:
            resp = self.idp.initiate_auth(
                ClientId=self.app_client_id,
                AuthFlow="USER_PASSWORD_AUTH",
                AuthParameters={"USERNAME": username, "PASSWORD": password},
            )
        except ClientError as e:
            # e.response['Error']['Code'] can be NotAuthorizedException, UserNotFoundException, etc.
            raise RuntimeError(e.response["Error"]["Message"])

        auth_result = resp.get("AuthenticationResult")
        if not auth_result:
            raise RuntimeError("Authentication failed: no tokens returned")

        return {
            "IdToken": auth_result.get("IdToken"),
            "AccessToken": auth_result.get("AccessToken"),
            "RefreshToken": auth_result.get("RefreshToken"),
            "ExpiresIn": auth_result.get("ExpiresIn"),
            "TokenType": auth_result.get("TokenType"),
            "IssuedAt": int(time.time()),
        }

    def refresh(self, refresh_token: str) -> Dict[str, Any]:
        """Use REFRESH_TOKEN_AUTH to get new Id/Access tokens."""
        try:
            resp = self.idp.initiate_auth(
                ClientId=self.app_client_id,
                AuthFlow="REFRESH_TOKEN_AUTH",
                AuthParameters={"REFRESH_TOKEN": refresh_token},
            )
        except ClientError as e:
            raise RuntimeError(e.response["Error"]["Message"])

        auth_result = resp.get("AuthenticationResult")
        if not auth_result:
            raise RuntimeError("Refresh failed: no tokens returned")

        return {
            "IdToken": auth_result.get("IdToken"),
            "AccessToken": auth_result.get("AccessToken"),
            # Note: AWS often does not return a new RefreshToken here
            "RefreshToken": refresh_token,
            "ExpiresIn": auth_result.get("ExpiresIn"),
            "TokenType": auth_result.get("TokenType"),
            "IssuedAt": int(time.time()),
        }

    def verify_id_token(self, id_token: str) -> Dict[str, Any]:
        """
        Verify and decode a Cognito IdToken (JWT). Returns the decoded claims if valid.
        """
        jwks = self._get_jwks()
        unverified_header = jwt.get_unverified_header(id_token)
        kid = unverified_header.get("kid")
        key = None
        for k in jwks["keys"]:
            if k["kid"] == kid:
                key = k
                break
        if not key:
            raise RuntimeError("Unable to find matching JWK for token kid")

        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key))
        issuer = f"https://cognito-idp.{self.region}.amazonaws.com/{self.user_pool_id}"

        # Validate signature, issuer, audience (client id)
        claims = jwt.decode(
            id_token,
            public_key,
            algorithms=["RS256"],
            audience=self.app_client_id,
            issuer=issuer,
        )
        return claims

    def is_token_expiring(self, issued_at: int, expires_in: int, skew_sec: int = 60) -> bool:
        """True if token is within skew_sec of expiry."""
        return time.time() >= (issued_at + expires_in - skew_sec)

    # ----- Internal helpers -----

    def _get_jwks(self) -> Dict[str, Any]:
        # Cache JWKS to avoid per-request network calls
        if self._jwks_cache and (time.time() - self._jwks_cache_ts) < self._jwks_ttl_sec:
            return self._jwks_cache
        url = f"https://cognito-idp.{self.region}.amazonaws.com/{self.user_pool_id}/.well-known/jwks.json"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        self._jwks_cache = r.json()
        self._jwks_cache_ts = time.time()
        return self._jwks_cache
