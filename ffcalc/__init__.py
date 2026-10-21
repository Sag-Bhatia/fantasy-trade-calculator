import truststore

# This Mac sits behind a TLS-inspecting proxy; trust the macOS keychain instead of certifi.
truststore.inject_into_ssl()
