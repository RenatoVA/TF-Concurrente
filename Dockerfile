FROM golang:1.24-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /taxi ./cmd/taxi

FROM alpine:3.19
RUN apk --no-cache add ca-certificates
COPY --from=builder /taxi /usr/local/bin/taxi
ENTRYPOINT ["taxi"]
