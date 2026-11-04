"""AWS dynamic client, listing, resolution, and search."""

from .allowlist import ALLOWED_SERVICES, is_operation_allowed, is_service_allowed
from .catalogue import get_service_info, list_catalogue
from .resource_lister import list_resources, supported_list_services

__all__ = [
    "ALLOWED_SERVICES",
    "is_service_allowed",
    "is_operation_allowed",
    "list_catalogue",
    "get_service_info",
    "list_resources",
    "supported_list_services",
]
