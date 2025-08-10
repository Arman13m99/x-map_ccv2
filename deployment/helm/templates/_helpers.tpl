{{/*
_helpers.tpl - Template helpers for Tapsi Food Map Dashboard Helm chart
Expand the name of the chart.
*/}}
{{- define "tapsi-food-map.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "tapsi-food-map.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "tapsi-food-map.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "tapsi-food-map.labels" -}}
helm.sh/chart: {{ include "tapsi-food-map.chart" . }}
{{ include "tapsi-food-map.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- with .Values.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "tapsi-food-map.selectorLabels" -}}
app.kubernetes.io/name: {{ include "tapsi-food-map.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "tapsi-food-map.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "tapsi-food-map.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create a default fully qualified app name for nginx.
*/}}
{{- define "tapsi-food-map.nginx.fullname" -}}
{{- printf "%s-nginx" (include "tapsi-food-map.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name for celery worker.
*/}}
{{- define "tapsi-food-map.celery.worker.fullname" -}}
{{- printf "%s-celery-worker" (include "tapsi-food-map.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name for celery beat.
*/}}
{{- define "tapsi-food-map.celery.beat.fullname" -}}
{{- printf "%s-celery-beat" (include "tapsi-food-map.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name for celery flower.
*/}}
{{- define "tapsi-food-map.celery.flower.fullname" -}}
{{- printf "%s-celery-flower" (include "tapsi-food-map.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name for postgresql.
*/}}
{{- define "tapsi-food-map.postgresql.fullname" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "%s-postgresql" (include "tapsi-food-map.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-postgresql" .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{/*
Create a default fully qualified app name for redis.
*/}}
{{- define "tapsi-food-map.redis.fullname" -}}
{{- if .Values.redis.enabled }}
{{- printf "%s-redis-master" (include "tapsi-food-map.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-redis" .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{/*
Create the PostgreSQL connection string
*/}}
{{- define "tapsi-food-map.postgresql.connectionString" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "postgresql://%s:%s@%s:5432/%s" .Values.postgresql.auth.username .Values.secrets.dbPassword (include "tapsi-food-map.postgresql.fullname" .) .Values.postgresql.auth.database }}
{{- else }}
{{- printf "postgresql://%s:%s@%s:5432/%s" .Values.config.database.username .Values.secrets.dbPassword .Values.externalPostgresql.host .Values.config.database.name }}
{{- end }}
{{- end }}

{{/*
Create the Redis connection string
*/}}
{{- define "tapsi-food-map.redis.connectionString" -}}
{{- if .Values.redis.enabled }}
{{- if .Values.redis.auth.enabled }}
{{- printf "redis://:%s@%s:6379/0" .Values.secrets.redisPassword (include "tapsi-food-map.redis.fullname" .) }}
{{- else }}
{{- printf "redis://%s:6379/0" (include "tapsi-food-map.redis.fullname" .) }}
{{- end }}
{{- else }}
{{- if .Values.externalRedis.password }}
{{- printf "redis://:%s@%s:%d/%d" .Values.externalRedis.password .Values.externalRedis.host .Values.externalRedis.port .Values.externalRedis.database }}
{{- else }}
{{- printf "redis://%s:%d/%d" .Values.externalRedis.host .Values.externalRedis.port .Values.externalRedis.database }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Get the Docker image name
*/}}
{{- define "tapsi-food-map.image" -}}
{{- $registry := .Values.global.imageRegistry | default .Values.app.image.registry -}}
{{- $repository := .Values.app.image.repository -}}
{{- $tag := .Values.app.image.tag | default .Chart.AppVersion -}}
{{- if $registry -}}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- else -}}
{{- printf "%s:%s" $repository $tag -}}
{{- end -}}
{{- end }}

{{/*
Get the nginx image name
*/}}
{{- define "tapsi-food-map.nginx.image" -}}
{{- $registry := .Values.global.imageRegistry | default .Values.nginx.image.registry -}}
{{- $repository := .Values.nginx.image.repository -}}
{{- $tag := .Values.nginx.image.tag -}}
{{- if $registry -}}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- else -}}
{{- printf "%s:%s" $repository $tag -}}
{{- end -}}
{{- end }}

{{/*
Create environment variables for the application
*/}}
{{- define "tapsi-food-map.env" -}}
- name: FLASK_ENV
  value: {{ .Values.config.flask.env | quote }}
- name: DEBUG
  value: {{ .Values.config.flask.debug | quote }}
- name: HOST
  value: {{ .Values.config.flask.host | quote }}
- name: PORT
  value: {{ .Values.config.flask.port | quote }}
- name: WORKERS
  value: {{ .Values.config.flask.workers | quote }}
- name: TIMEOUT
  value: {{ .Values.config.flask.timeout | quote }}
- name: DATABASE_URL
  value: {{ include "tapsi-food-map.postgresql.connectionString" . | quote }}
- name: REDIS_URL
  value: {{ include "tapsi-food-map.redis.connectionString" . | quote }}
- name: SECRET_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "tapsi-food-map.fullname" . }}-secrets
      key: secretKey
- name: METABASE_URL
  value: {{ .Values.config.metabase.url | quote }}
- name: METABASE_USERNAME
  valueFrom:
    secretKeyRef:
      name: {{ include "tapsi-food-map.fullname" . }}-secrets
      key: metabaseUsername
- name: METABASE_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "tapsi-food-map.fullname" . }}-secrets
      key: metabasePassword
- name: METABASE_TEAM
  value: {{ .Values.config.metabase.team | quote }}
- name: ORDER_DATA_QUESTION_ID
  value: {{ .Values.config.metabase.orderDataQuestionId | quote }}
- name: VENDOR_DATA_QUESTION_ID
  value: {{ .Values.config.metabase.vendorDataQuestionId | quote }}
- name: CACHE_TTL
  value: {{ .Values.config.cache.ttl | quote }}
- name: CHUNK_SIZE
  value: {{ .Values.config.cache.chunkSize | quote }}
- name: WORKER_COUNT
  value: {{ .Values.config.cache.workerCount | quote }}
- name: PAGE_SIZE
  value: {{ .Values.config.cache.pageSize | quote }}
- name: LOG_LEVEL
  value: "info"
- name: ENVIRONMENT
  value: {{ .Release.Namespace | quote }}
- name: VERSION
  value: {{ .Chart.AppVersion | quote }}
- name: CORS_ORIGINS
  value: {{ .Values.config.security.corsOrigins | quote }}
- name: TRUSTED_PROXIES
  value: {{ .Values.config.security.trustedProxies | quote }}
{{- end }}

{{/*
Resource limits and requests
*/}}
{{- define "tapsi-food-map.resources" -}}
{{- with .Values.app.resources }}
resources:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Image pull secrets
*/}}
{{- define "tapsi-food-map.imagePullSecrets" -}}
{{- with .Values.global.imagePullSecrets }}
imagePullSecrets:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Security context
*/}}
{{- define "tapsi-food-map.securityContext" -}}
{{- with .Values.app.securityContext }}
securityContext:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Node selector
*/}}
{{- define "tapsi-food-map.nodeSelector" -}}
{{- with .Values.nodeSelector }}
nodeSelector:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Tolerations
*/}}
{{- define "tapsi-food-map.tolerations" -}}
{{- with .Values.tolerations }}
tolerations:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Affinity
*/}}
{{- define "tapsi-food-map.affinity" -}}
{{- with .Values.affinity }}
affinity:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Pod annotations
*/}}
{{- define "tapsi-food-map.podAnnotations" -}}
{{- with .Values.podAnnotations }}
annotations:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}