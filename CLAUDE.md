# CLAUDE.md - Tapsi Food Map Dashboard Production Optimization

## Project Overview
The Tapsi Food Map Dashboard is a Flask-based web application that visualizes vendor locations, order density heatmaps, and coverage analytics using real-time data from Metabase. The current implementation loads all data on startup and serves it through REST APIs to a JavaScript frontend.

## Current Architecture Issues
1. **Data Loading**: All data (3M+ order records) loaded synchronously on startup
2. **Memory Usage**: Large DataFrames held in memory continuously
3. **Concurrent Access**: No optimization for multiple simultaneous users
4. **Data Freshness**: Manual refresh required for updated data
5. **Performance**: Heavy computational operations blocking request handling

## Production Optimization Goals

### 1. DATA MANAGEMENT & CACHING STRATEGY

#### Priority: CRITICAL
- **df_orders**: 3M+ rows, fetch once daily at 9am (except first run)
- **df_vendors**: Fetch every 10 minutes
- Implement intelligent caching to prevent server crashes
- Optimize memory usage for concurrent users

#### Implementation Tasks:
- [ ] **Background Data Scheduler**
  - [ ] Implement APScheduler for automated data fetching
  - [ ] Create daily job for df_orders (9am UTC/local)
  - [ ] Create 10-minute interval job for df_vendors
  - [ ] Add job persistence and recovery mechanisms
  
- [ ] **Memory-Efficient Data Storage**
  - [ ] Implement DataFrame chunking for large datasets
  - [ ] Use Pandas categorical dtypes for memory optimization
  - [ ] Implement data compression (Parquet format)
  - [ ] Create data versioning system
  
- [ ] **Intelligent Caching Layer**
  - [ ] Redis/Memcached integration for processed results
  - [ ] Cache heatmap calculations by parameters
  - [ ] Implement cache invalidation strategies
  - [ ] Add cache warming for common queries

### 2. CONCURRENCY & SCALABILITY

#### Priority: HIGH
- Support multiple simultaneous users
- Prevent blocking operations
- Implement request queuing and load balancing

#### Implementation Tasks:
- [ ] **Async Processing Architecture**
  - [ ] Convert heavy computations to async background tasks
  - [ ] Implement Celery with Redis/RabbitMQ
  - [ ] Create task queue for heatmap generation
  - [ ] Add progress tracking for long-running operations
  
- [ ] **Database Integration**
  - [ ] Replace in-memory DataFrames with database queries
  - [ ] Implement PostgreSQL with PostGIS for spatial data
  - [ ] Create indexed views for common aggregations
  - [ ] Add connection pooling and query optimization
  
- [ ] **Request Optimization**
  - [ ] Implement request debouncing on frontend
  - [ ] Add pagination for large result sets
  - [ ] Create streaming responses for large data
  - [ ] Implement request rate limiting

### 3. APPLICATION ARCHITECTURE REFACTORING

#### Priority: HIGH
- Modular, maintainable codebase
- Separation of concerns
- Error handling and monitoring

#### Implementation Tasks:
- [ ] **Code Restructuring**
  - [ ] Split app.py into modules (models, services, routes)
  - [ ] Create data access layer (DAL)
  - [ ] Implement service layer for business logic
  - [ ] Add dependency injection container
  
- [ ] **Configuration Management**
  - [ ] Environment-based configuration
  - [ ] Secret management (database credentials)
  - [ ] Feature flags for A/B testing
  - [ ] Centralized logging configuration
  
- [ ] **Error Handling & Monitoring**
  - [ ] Comprehensive exception handling
  - [ ] Health check endpoints
  - [ ] Application metrics (Prometheus/Grafana)
  - [ ] Error tracking (Sentry integration)

### 4. PERFORMANCE OPTIMIZATION

#### Priority: MEDIUM-HIGH
- Frontend and backend performance improvements
- Caching strategies
- Code optimization

#### Implementation Tasks:
- [ ] **Frontend Optimization**
  - [ ] Implement lazy loading for map components
  - [ ] Add service worker for offline capability
  - [ ] Optimize bundle size and loading
  - [ ] Implement progressive web app features
  
- [ ] **Backend Performance**
  - [ ] Profile and optimize hot code paths
  - [ ] Implement query result caching
  - [ ] Add database query optimization
  - [ ] Use compiled extensions (NumPy/Pandas C extensions)
  
- [ ] **API Optimization**
  - [ ] Implement GraphQL for flexible data fetching
  - [ ] Add response compression (gzip)
  - [ ] Implement ETag caching
  - [ ] Create API versioning strategy

### 5. DEPLOYMENT & INFRASTRUCTURE

#### Priority: HIGH
- Production-ready deployment
- Scalability and reliability
- Security considerations

#### Implementation Tasks:
- [ ] **Containerization**
  - [ ] Create optimized Dockerfile
  - [ ] Docker Compose for local development
  - [ ] Multi-stage builds for production
  - [ ] Health checks and graceful shutdowns
  
- [ ] **Kubernetes Deployment**
  - [ ] Create Kubernetes manifests
  - [ ] Implement horizontal pod autoscaling
  - [ ] Add persistent volumes for data
  - [ ] Configure ingress and load balancing
  
- [ ] **CI/CD Pipeline**
  - [ ] GitHub Actions/GitLab CI setup
  - [ ] Automated testing and linting
  - [ ] Security scanning
  - [ ] Automated deployment strategies
  
- [ ] **Security Implementation**
  - [ ] Add authentication and authorization
  - [ ] Implement CORS properly
  - [ ] Add rate limiting and DDoS protection
  - [ ] Secure API endpoints

### 6. MONITORING & OBSERVABILITY

#### Priority: MEDIUM
- Production monitoring and alerting
- Performance tracking
- User analytics

#### Implementation Tasks:
- [ ] **Application Monitoring**
  - [ ] Add structured logging (JSON format)
  - [ ] Implement distributed tracing
  - [ ] Create custom metrics for business logic
  - [ ] Set up alerting rules
  
- [ ] **Infrastructure Monitoring**
  - [ ] Server resource monitoring
  - [ ] Database performance monitoring
  - [ ] Network and external service monitoring
  - [ ] Log aggregation and analysis

## IMPLEMENTATION PHASES

### Phase 1: Critical Stability (Week 1-2)
1. Implement background data scheduler
2. Add basic caching layer
3. Convert blocking operations to async
4. Basic error handling and logging

### Phase 2: Scalability Foundation (Week 3-4)
1. Database integration
2. Request optimization
3. Code restructuring
4. Basic monitoring

### Phase 3: Production Deployment Preparation (Week 5-6)
1. Complete containerization with production configs
2. Full CI/CD pipeline setup and testing
3. Security implementation and hardening
4. Performance optimization and validation
5. Create comprehensive deployment documentation

### Phase 4: Deployment Ready Package (Week 7-8)
1. Final integration testing
2. Production environment configuration
3. Deployment automation scripts
4. Handover documentation and runbooks
5. Performance benchmarking reports

## CURRENT PROJECT ANALYSIS

### Existing Codebase Structure
```
tapsi-food-map/
├── app.py (1,200+ lines - NEEDS REFACTORING)
├── mini.py (Metabase data fetcher)
├── public/
│   ├── index.html (Frontend dashboard)
│   ├── script.js (Map visualization logic)
│   └── styles.css (UI styling)
├── src/
│   ├── polygons/ (Shapefiles for areas)
│   ├── vendor/ (Grade data)
│   └── targets/ (Coverage targets)
└── requirements.txt
```

### Current Pain Points Identified
1. **app.py Line 523-600**: Synchronous data loading blocks startup
2. **Memory Issue**: `df_orders` (3M rows) loaded entirely in memory
3. **Line 890-950**: Heavy heatmap calculations block request threads
4. **No error recovery**: Metabase failures crash the app
5. **No data persistence**: All data lost on restart

### Critical Functions to Refactor
- `load_data()` - Convert to async background task
- `get_map_data()` - Add caching and pagination
- `generate_improved_heatmap_data()` - Move to background worker
- `calculate_coverage_for_grid_vectorized()` - Optimize memory usage

## DETAILED CODE CHANGES CHECKLIST

### PHASE 1: IMMEDIATE STABILITY (Day 1-3)

#### File: `config.py` (NEW)
- [ ] Environment-based configuration management
- [ ] Database connection settings
- [ ] Cache configuration
- [ ] Logging setup
- [ ] Metabase credentials management

#### File: `scheduler.py` (NEW)
- [ ] APScheduler integration
- [ ] Vendor data fetching job (10-minute interval)
- [ ] Order data fetching job (daily 9am)
- [ ] Job persistence and recovery
- [ ] Error handling and retry logic

#### File: `cache.py` (NEW)
- [ ] Redis connection management
- [ ] Cache key strategies
- [ ] TTL management
- [ ] Cache invalidation
- [ ] Fallback mechanisms

#### File: `models.py` (NEW)
- [ ] Data models for vendors, orders, polygons
- [ ] Database schema definitions
- [ ] Data validation
- [ ] Serialization methods

#### Modify: `app.py`
- [ ] Remove synchronous data loading from startup
- [ ] Add health check endpoint `/health`
- [ ] Implement basic error handling
- [ ] Add request logging middleware
- [ ] Convert heavy endpoints to async

### PHASE 2: ARCHITECTURE REFACTORING (Week 1)

#### File: `services/data_service.py` (NEW)
- [ ] Data fetching and processing logic
- [ ] Chunk processing for large datasets
- [ ] Data validation and cleaning
- [ ] Error recovery mechanisms

#### File: `services/map_service.py` (NEW)
- [ ] Heatmap generation logic
- [ ] Coverage grid calculations
- [ ] Polygon enrichment
- [ ] Spatial operations

#### File: `services/cache_service.py` (NEW)
- [ ] High-level caching operations
- [ ] Cache warming strategies
- [ ] Cache statistics
- [ ] Performance monitoring

#### File: `database.py` (NEW)
- [ ] Database connection management
- [ ] Migration scripts
- [ ] Query optimization
- [ ] Connection pooling

#### File: `celery_app.py` (NEW)
- [ ] Celery configuration
- [ ] Task definitions
- [ ] Result backend
- [ ] Monitoring setup

### PHASE 3: PRODUCTION READINESS (Week 2)

#### File: `Dockerfile` (NEW)
- [ ] Multi-stage build for optimization
- [ ] Security hardening
- [ ] Health checks
- [ ] Proper user permissions

#### File: `docker-compose.yml` (NEW)
- [ ] Application container
- [ ] Redis container
- [ ] PostgreSQL container
- [ ] Nginx proxy

#### File: `deployment/` (NEW DIRECTORY)
- [ ] Complete Kubernetes deployment manifests
- [ ] Helm charts for easy deployment
- [ ] Environment-specific configurations
- [ ] Load balancer and ingress setup
- [ ] SSL/TLS certificate management
- [ ] Resource scaling configurations

#### File: `scripts/` (NEW DIRECTORY)
- [ ] Deployment automation scripts
- [ ] Database migration runner
- [ ] Environment setup scripts
- [ ] Health check validation scripts
- [ ] Performance testing automation

#### File: `docs/` (NEW DIRECTORY)
- [ ] Deployment guide (step-by-step)
- [ ] Architecture documentation
- [ ] API documentation
- [ ] Troubleshooting guide
- [ ] Performance tuning guide
- [ ] Security hardening checklist

#### File: `.github/workflows/` (NEW DIRECTORY)
- [ ] CI/CD pipeline
- [ ] Testing automation
- [ ] Security scanning
- [ ] Deployment automation

## SPECIFIC IMPLEMENTATION GUIDANCE

### Critical Code Patterns to Implement

#### 1. Data Loading Pattern
```python
# Replace current synchronous loading with:
@app.before_first_request
def initialize_app():
    scheduler.start()
    # Trigger initial data load asynchronously
    load_initial_data.delay()
```

#### 2. Caching Pattern
```python
# Add to all heavy computation endpoints:
@cache.memoize(timeout=3600)
def expensive_calculation(params):
    # computation logic
    pass
```

#### 3. Error Handling Pattern
```python
# Wrap all external API calls:
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_metabase_data():
    # API call logic
    pass
```

### Memory Optimization Strategies

#### DataFrame Chunking Implementation
```python
def process_large_dataframe(df, chunk_size=100000):
    for chunk in pd.read_csv(source, chunksize=chunk_size):
        yield process_chunk(chunk)
```

#### Memory-Efficient Data Types
```python
# Convert object columns to categories
for col in categorical_columns:
    df[col] = df[col].astype('category')
```

### Database Migration Strategy

#### 1. Create Migration Scripts
- [ ] `migrations/001_initial_schema.sql`
- [ ] `migrations/002_add_indexes.sql`
- [ ] `migrations/003_add_partitions.sql`

#### 2. Data Transfer Process
- [ ] Migrate historical order data in batches
- [ ] Create materialized views for common queries
- [ ] Set up real-time sync with Metabase

## TESTING REQUIREMENTS

### Load Testing Scripts (NEW)
- [ ] `tests/load_test.py` - Simulate 50+ concurrent users
- [ ] `tests/memory_test.py` - Monitor memory usage under load
- [ ] `tests/data_integrity_test.py` - Verify data consistency

### Integration Tests
- [ ] `tests/test_scheduler.py` - Background job testing
- [ ] `tests/test_cache.py` - Cache functionality
- [ ] `tests/test_api.py` - API endpoint testing

## MONITORING & OBSERVABILITY

### Metrics to Track
- [ ] Request latency by endpoint
- [ ] Memory usage per process
- [ ] Cache hit/miss ratios
- [ ] Background job success rates
- [ ] Database query performance

### Alerting Rules
- [ ] High memory usage (>80%)
- [ ] Failed background jobs
- [ ] API response time >2s
- [ ] Cache miss rate >20%

## PRODUCTION DEPLOYMENT AUTOMATION

### Complete Deployment Package Structure
```
tapsi-food-map-production/
├── app/ (Optimized application code)
├── deployment/
│   ├── kubernetes/
│   │   ├── namespace.yaml
│   │   ├── configmap.yaml
│   │   ├── secrets.yaml (template)
│   │   ├── postgres-deployment.yaml
│   │   ├── redis-deployment.yaml
│   │   ├── app-deployment.yaml
│   │   ├── celery-deployment.yaml
│   │   ├── nginx-deployment.yaml
│   │   ├── services.yaml
│   │   ├── ingress.yaml
│   │   └── hpa.yaml (horizontal pod autoscaler)
│   ├── helm/
│   │   ├── Chart.yaml
│   │   ├── values.yaml
│   │   ├── values-prod.yaml
│   │   ├── values-staging.yaml
│   │   └── templates/ (Helm templates)
│   ├── terraform/ (Infrastructure as Code)
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── environments/
│   └── ansible/ (Configuration management)
├── scripts/
│   ├── deploy.sh (One-click deployment)
│   ├── setup-environment.sh
│   ├── run-migrations.sh
│   ├── health-check.sh
│   ├── load-test.sh
│   └── rollback.sh
├── monitoring/
│   ├── prometheus/ (Monitoring configs)
│   ├── grafana/ (Dashboard configs)
│   └── alertmanager/ (Alert configs)
└── docs/
    ├── DEPLOYMENT.md
    ├── ARCHITECTURE.md
    ├── TROUBLESHOOTING.md
    └── HANDOVER.md
```

### Automated Deployment Script
```bash
#!/bin/bash
# deploy.sh - One-click production deployment

set -e

echo "🚀 Starting Tapsi Food Map Production Deployment"

# Validate environment
./scripts/validate-environment.sh

# Setup infrastructure
if [ "$INFRASTRUCTURE_PROVIDER" = "aws" ]; then
    cd deployment/terraform/aws && terraform apply -auto-approve
elif [ "$INFRASTRUCTURE_PROVIDER" = "gcp" ]; then
    cd deployment/terraform/gcp && terraform apply -auto-approve
fi

# Deploy application
if [ "$DEPLOYMENT_METHOD" = "helm" ]; then
    helm upgrade --install tapsi-food-map ./deployment/helm \
        -f ./deployment/helm/values-prod.yaml \
        --namespace tapsi-food-map
else
    kubectl apply -f ./deployment/kubernetes/
fi

# Run health checks
./scripts/health-check.sh

echo "✅ Deployment completed successfully!"
echo "📊 Access dashboard at: https://tapsi-food-map.yourdomain.com"
```

### Environment Configuration Templates
```yaml
# deployment/helm/values-prod.yaml
app:
  image:
    repository: tapsi-food-map
    tag: "{{ .Values.global.imageTag }}"
  replicas: 3
  resources:
    requests:
      memory: "1Gi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "1000m"

database:
  enabled: true
  persistence:
    size: 100Gi
  resources:
    requests:
      memory: "2Gi"
      cpu: "1000m"

redis:
  enabled: true
  persistence:
    size: 10Gi

ingress:
  enabled: true
  hostname: tapsi-food-map.yourdomain.com
  tls:
    enabled: true
```

### Complete CI/CD Pipeline
```yaml
# .github/workflows/production-deployment.yml
name: Production Deployment

on:
  push:
    tags:
      - 'v*'

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
          
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
          
      - name: Run tests
        run: |
          pytest tests/ --cov=app --cov-report=xml
          
      - name: Security scan
        run: |
          bandit -r app/
          safety check
          
      - name: Build Docker image
        run: |
          docker build -t tapsi-food-map:${{ github.sha }} .
          
      - name: Run integration tests
        run: |
          docker-compose -f docker-compose.test.yml up --abort-on-container-exit
          
  deploy-to-staging:
    needs: build-and-test
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: Deploy to staging
        run: |
          ./scripts/deploy.sh staging
          
      - name: Run E2E tests
        run: |
          ./scripts/e2e-tests.sh staging
          
  deploy-to-production:
    needs: deploy-to-staging
    runs-on: ubuntu-latest
    environment: production
    if: startsWith(github.ref, 'refs/tags/v')
    steps:
      - name: Deploy to production
        run: |
          ./scripts/deploy.sh production
          
      - name: Verify deployment
        run: |
          ./scripts/health-check.sh production
          ./scripts/smoke-tests.sh production
```

## TECHNICAL SPECIFICATIONS

### Memory Optimization
```python
# Target memory usage per user session
MAX_MEMORY_PER_SESSION = "500MB"
CHUNK_SIZE = 100000  # Process data in chunks
CACHE_TTL = 3600  # 1 hour cache for processed results
```

### Performance Targets
- **Page Load Time**: < 2 seconds
- **API Response Time**: < 500ms for cached, < 3s for computed
- **Concurrent Users**: 50+ without degradation
- **Data Refresh**: Vendors 10min, Orders daily at 9am

### Caching Strategy
```python
# Cache layers
LEVEL_1 = "Redis (Hot data, 1-hour TTL)"
LEVEL_2 = "Database views (Warm data, 1-day TTL)"
LEVEL_3 = "File system (Cold data, 7-day TTL)"
```

## TESTING STRATEGY

### Load Testing
- [ ] Test with 50+ concurrent users
- [ ] Stress test with 3M+ order records
- [ ] Memory usage under load
- [ ] Cache performance testing

### Integration Testing
- [ ] End-to-end API testing
- [ ] Database integration testing
- [ ] Frontend-backend integration
- [ ] External service dependency testing

### Performance Testing
- [ ] Benchmark critical code paths
- [ ] Memory profiling
- [ ] Database query optimization
- [ ] Frontend loading performance

## DEPLOYMENT READINESS CHECKLIST

### Pre-deployment Validation
- [ ] All automated tests passing (unit, integration, e2e)
- [ ] Security scans completed with no critical issues
- [ ] Performance benchmarks meeting requirements
- [ ] Database migration scripts tested
- [ ] Environment configurations validated
- [ ] SSL certificates prepared
- [ ] DNS configurations ready
- [ ] Monitoring dashboards configured
- [ ] Alerting rules tested
- [ ] Rollback procedures verified

### Deployment Package Contents
- [ ] **Complete application code** (optimized and production-ready)
- [ ] **Docker images** (built and security-scanned)
- [ ] **Kubernetes manifests** (environment-specific)
- [ ] **Helm charts** (for easy deployment)
- [ ] **Terraform/Ansible scripts** (infrastructure automation)
- [ ] **CI/CD pipeline** (fully configured and tested)
- [ ] **Deployment scripts** (one-click deployment)
- [ ] **Monitoring setup** (Prometheus, Grafana, alerts)
- [ ] **Database migrations** (automated and tested)
- [ ] **Environment templates** (dev, staging, prod)

### Documentation Package
- [ ] **Step-by-step deployment guide**
- [ ] **Architecture documentation**
- [ ] **API documentation** (OpenAPI/Swagger)
- [ ] **Environment setup guide**
- [ ] **Troubleshooting manual**
- [ ] **Performance tuning guide**
- [ ] **Security hardening checklist**
- [ ] **Monitoring and alerting guide**
- [ ] **Backup and recovery procedures**
- [ ] **Incident response playbook**

### Handover Deliverables
- [ ] **Complete codebase** in production-ready state
- [ ] **Deployment automation** (infrastructure + application)
- [ ] **Monitoring and observability** fully configured
- [ ] **Security hardening** implemented and tested
- [ ] **Performance optimization** validated with load tests
- [ ] **Documentation** comprehensive and up-to-date
- [ ] **CI/CD pipeline** functional and tested
- [ ] **Environment configurations** for all stages
- [ ] **Support runbooks** for operations team
- [ ] **Training materials** for maintenance team

## FINAL DELIVERY PACKAGE

### What You'll Receive (100% Production Ready)
```
📦 COMPLETE DEPLOYMENT PACKAGE
├── 🚀 Optimized Application (handles 50+ users, 3M+ records)
├── 🐳 Production Docker Images (multi-stage, security-hardened)
├── ☸️ Kubernetes Deployment (auto-scaling, health checks)
├── 📊 Monitoring Stack (Prometheus + Grafana + alerts)
├── 🔄 CI/CD Pipeline (GitHub Actions, automated testing)
├── 🛡️ Security Hardening (secrets management, RBAC)
├── 📚 Complete Documentation (deployment + operations)
├── 🧪 Testing Suite (unit + integration + load tests)
├── 📈 Performance Validation (benchmarks + optimization)
└── 🔧 Deployment Automation (one-click deployment)
```

### Deployment Team Responsibilities (After Handover)
1. **Infrastructure Setup** (5-10 minutes)
   - Run terraform/ansible scripts for cloud resources
   - Configure DNS and load balancers
   - Set up SSL certificates

2. **Application Deployment** (5-10 minutes)
   - Execute deployment scripts
   - Configure environment-specific variables
   - Verify health checks

3. **Go-Live Validation** (15-30 minutes)
   - Run smoke tests
   - Verify monitoring dashboards
   - Validate performance metrics
   - Test alerting systems

### Performance Guarantees
- ✅ **Response Time**: <500ms for cached requests, <3s for computed
- ✅ **Concurrent Users**: 50+ without performance degradation
- ✅ **Memory Efficiency**: <2GB per application instance
- ✅ **Data Freshness**: Vendors 10min, Orders daily at 9am
- ✅ **Uptime**: 99.9% with proper infrastructure
- ✅ **Auto-scaling**: Horizontal scaling based on CPU/memory

## MAINTENANCE TASKS

### Daily
- [ ] Check application health
- [ ] Monitor error rates
- [ ] Review performance metrics
- [ ] Verify data freshness

### Weekly
- [ ] Review logs for patterns
- [ ] Performance optimization review
- [ ] Security updates check
- [ ] Backup verification

### Monthly
- [ ] Capacity planning review
- [ ] Security audit
- [ ] Performance tuning
- [ ] Documentation updates

## SUCCESS METRICS

### Performance
- API response time < 500ms (90th percentile)
- Page load time < 2 seconds
- 99.9% uptime
- Support 50+ concurrent users with auto-scaling
- Comprehensive monitoring and alerting setup
- Complete deployment automation and documentation