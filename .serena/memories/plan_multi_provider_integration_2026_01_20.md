# Multi-Provider Financial Integration Plan

**Created**: 2026-01-20
**Status**: Active
**Owner**: AI Agent

## Executive Summary

Implementation of complete Yapily API integration, enhancement of NormalizationService, and addition of Coinbase and Binance cryptocurrency providers following a 4-phase sequential approach.

## Current State Analysis

### Yapily Provider (Scaffold Complete)
- ✅ Basic provider class structure exists (`YapilyProvider`)
- ✅ Extends `FinancialDataProvider` base class
- ✅ Mock implementations for `getAccounts()` and `getTransactions()`
- ❌ No real OAuth 2.0 authentication flow
- ❌ No real API endpoint integration
- ❌ No token management or refresh mechanism

### NormalizationService (Basic Implementation)
- ✅ Basic transaction normalization exists
- ✅ MCC mapping defined in `/src/types/financial.ts` (limited coverage)
- ✅ Exchange rate config exists (`/src/config/exchange-rate.ts`)
- ❌ Currency conversion hardcoded to EUR only (TODO comment)
- ❌ No real-time exchange rate API integration
- ❌ No crypto currency support
- ❌ Limited MCC coverage (~50 codes)

### Provider Framework
- ✅ Abstract base class `FinancialDataProvider`
- ✅ Common interfaces: `ProviderAccount`, `ProviderTransaction`
- ✅ Plaid provider as reference implementation
- ✅ Integration with `TransactionSyncService`

## Phase 1: Yapily API Integration Completion

### Task 1.1: OAuth 2.0 Authentication Flow

**API Documentation Summary** (from Context7):
- **Authorization Request**: POST `/account-auth-requests`
  - Requires: `userUuid`, `institutionId`, `callbackUrl`
  - Returns: `authorisationUrl`, `status`
- **Token Exchange**: POST `/consent-auth-code`
  - Requires: `authCode`, `authState`
  - Returns: `consentToken`, `idToken`
- **Callback Handling**: GET on callback URL
  - Receives: `consentToken`, `state` as query params

**Implementation Steps**:
1. Create `/src/providers/yapily/auth.service.ts`
   - `initiateAuthorization(userId, institutionId, callbackUrl)`
   - `handleCallback(authCode, authState)`
   - `refreshToken(consentToken)` (if supported)
   - Token storage in database (secure)

2. Create database schema for Yapily tokens
   - Add `yapilyConsentToken` field to PlaidItem or create YapilyToken table
   - Store: `consentToken`, `idToken`, `expiresAt`, `institutionId`

3. Environment configuration
   - `YAPILY_APPLICATION_KEY`
   - `YAPILY_APPLICATION_SECRET`
   - `YAPILY_BASE_URL` (default: https://api.yapily.com)
   - `YAPILY_CALLBACK_URL`

### Task 1.2: Real Yapily Endpoints

**API Documentation Summary**:
- **GET /accounts**: Returns array of accounts with balances
  - Headers: `Consent: {consentToken}`, Basic auth with app credentials
  - Response: `accounts[]` with `account_id`, `balances`, `account_details`

- **GET /accounts/{accountId}/transactions**: Returns transaction history
  - Headers: `Consent: {consentToken}`, Basic auth
  - Response: `data[]` with transaction details, pagination metadata

**Implementation Steps**:
1. Update `/src/providers/yapily/yapily.provider.ts`
   - Replace mock `getAccounts()` with real API call
   - Replace mock `getTransactions()` with real API call
   - Add error handling and retry logic
   - Add rate limiting (respect Yapily limits)

2. Create `/src/providers/yapily/client.ts`
   - HTTP client with Basic auth
   - Request/response interceptors
   - Error mapping (Yapily errors → standard errors)

3. Map Yapily responses to `ProviderAccount` and `ProviderTransaction`
   - Handle Yapily-specific fields
   - Normalize currency codes
   - Handle pagination for transactions

### Task 1.3: Testing & Validation

1. Create integration tests (`/tests/providers/yapily/integration.test.ts`)
2. Test with Yapily sandbox environment
3. Validate data normalization pipeline
4. Test error scenarios (expired tokens, API failures)

**Success Criteria**:
- ✓ OAuth flow completes successfully
- ✓ Can retrieve real accounts and transactions
- ✓ Data normalizes correctly through existing pipeline
- ✓ Error handling covers common failures
- ✓ Integration tests pass

---

## Phase 2: NormalizationService Enhancement

### Task 2.1: Currency Conversion System

**Requirements**:
- Support fiat currencies (EUR, USD, GBP, NOK, etc.)
- Support crypto currencies (BTC, ETH, USDT, etc.)
- Real-time rates with caching
- Fallback to cached/mock rates on API failure
- Performance target: <50ms overhead

**Implementation Steps**:
1. Create `/src/services/exchange-rate/exchange-rate.service.ts`
   - `getRate(from: string, to: string, timestamp?: Date)`
   - `convertAmount(amount, from, to, timestamp?)`
   - Integration with Fixer.io API (config already exists)
   - Redis caching layer (config already exists)
   - Fallback to mock rates

2. Add crypto exchange rate provider
   - Research: CoinGecko API or CryptoCompare API
   - Support BTC, ETH, USDT, BNB, etc.
   - Cache crypto rates (more volatile, shorter TTL)

3. Update `NormalizationService.normalizeTransaction()`
   - Replace hardcoded EUR conversion
   - Call `exchangeRateService.convertAmount()`
   - Store exchange rate and timestamp
   - Handle conversion errors gracefully

4. Create `/tests/services/exchange-rate/exchange-rate.service.test.ts`
   - Test fiat conversions
   - Test crypto conversions
   - Test caching behavior
   - Test fallback mechanisms

### Task 2.2: MCC Mapping Enhancement

**Current State**: ~50 MCC codes in `/src/types/financial.ts`
**Target**: 95%+ coverage of common transaction categories

**Implementation Steps**:
1. Research ISO 18245 MCC standard
   - Download comprehensive MCC database
   - Map to existing transaction categories

2. Create `/src/services/mcc/mcc-mapping.service.ts`
   - `getMCCCategory(mccCode: string): string`
   - `getMCCDescription(mccCode: string): string`
   - Hierarchical mapping: specific → category → default
   - Handle crypto transactions (no standard MCC)

3. Expand MCC database in `/src/types/financial.ts` or separate file
   - Add 200+ common MCC codes
   - Group by category
   - Add descriptions for clarity

4. Handle crypto transaction categorization
   - Custom logic for crypto exchanges
   - Merchant name pattern matching
   - Transaction type inference (buy, sell, transfer)

5. Update `NormalizationService` to use MCC service
   - Integrate MCC lookup
   - Store MCC code in normalized transaction
   - Fallback to description-based categorization

**Success Criteria**:
- ✓ Fiat + crypto conversion works
- ✓ 95%+ MCC coverage
- ✓ Backward compatible with existing Plaid integration
- ✓ Performance maintained (<50ms overhead)
- ✓ Comprehensive test coverage

---

## Phase 3: Coinbase Provider Implementation

### Task 3.1: Provider Scaffold & Authentication

**API Documentation Summary** (from Context7):
- **Authentication**: API Key + Passphrase + Signature
  - Headers: `cb-access-key`, `cb-access-passphrase`, `cb-access-sign`, `cb-access-timestamp`
  - Signature: HMAC SHA256 of timestamp + method + path + body
- **Alternative**: Client API Key for read-only (simpler)
- **OAuth**: Supported for user authorization flows

**Implementation Steps**:
1. Create `/src/providers/coinbase/coinbase.provider.ts`
   - Extend `FinancialDataProvider`
   - Implement `getAccounts()`
   - Implement `getTransactions()`

2. Create `/src/providers/coinbase/auth.service.ts`
   - API key signature generation
   - OAuth flow (if needed for user connections)
   - Token management

3. Create `/src/providers/coinbase/client.ts`
   - HTTP client with signature authentication
   - Request signing logic
   - Error handling

4. Environment configuration
   - `COINBASE_API_KEY`
   - `COINBASE_API_SECRET`
   - `COINBASE_API_PASSPHRASE`
   - `COINBASE_BASE_URL`

### Task 3.2: Endpoint Implementation

**API Endpoints**:
- **GET /coinbase-accounts**: List user wallets
  - Returns: `id`, `name`, `balance`, `currency`, `type`
- **GET /accounts/{id}/transactions**: Account transaction history
  - Returns: transaction details with crypto-specific fields

**Implementation Steps**:
1. Map Coinbase accounts to `ProviderAccount`
   - Handle crypto currencies
   - Convert balances to common format
   - Store crypto-specific metadata

2. Map Coinbase transactions to `ProviderTransaction`
   - Handle buy/sell/transfer types
   - Store crypto amount + fiat amount
   - Capture fees and exchange rates
   - Handle pending vs completed status

3. Add crypto-specific fields to `ProviderTransaction` interface
   - `cryptoAmount?: number`
   - `cryptoCurrency?: string`
   - `transactionType?: 'buy' | 'sell' | 'transfer' | 'receive'`
   - `networkFee?: number`

### Task 3.3: Integration & Testing

1. Integrate with enhanced `NormalizationService`
2. Test crypto currency conversion
3. Validate categorization of crypto transactions
4. Create integration tests with Coinbase sandbox

**Success Criteria**:
- ✓ Authentication works (API key or OAuth)
- ✓ Crypto accounts retrieval functional
- ✓ Transaction normalization correct
- ✓ Tests pass

---

## Phase 4: Binance Provider Implementation

### Task 4.1: Provider Scaffold & Authentication

**API Documentation Summary** (from Context7):
- **Authentication**: API Key + Secret with RSA signature
  - Header: `X-MBX-APIKEY`
  - Query param: `signature` (HMAC SHA256 or RSA)
  - Timestamp required in all requests
- **Signature**: Sign query string with private key

**Implementation Steps**:
1. Create `/src/providers/binance/binance.provider.ts`
   - Extend `FinancialDataProvider`
   - Implement `getAccounts()`
   - Implement `getTransactions()`

2. Create `/src/providers/binance/auth.service.ts`
   - RSA signature generation
   - HMAC SHA256 signature (alternative)
   - Timestamp management

3. Create `/src/providers/binance/client.ts`
   - HTTP client with signature authentication
   - Request signing logic
   - Error handling

4. Environment configuration
   - `BINANCE_API_KEY`
   - `BINANCE_API_SECRET`
   - `BINANCE_BASE_URL` (default: https://api.binance.com)

### Task 4.2: Endpoint Implementation

**API Endpoints**:
- **GET /api/v3/account**: Account information
  - Returns: balances array with `asset`, `free`, `locked`
  - Includes permissions and commission rates

**Implementation Steps**:
1. Map Binance account to `ProviderAccount`
   - Aggregate all balances into accounts
   - Handle spot vs futures separately
   - Convert crypto balances

2. Implement transaction history
   - Binance doesn't have unified transaction endpoint
   - May need to aggregate from multiple sources:
     - Trade history
     - Deposit history
     - Withdrawal history

3. Handle Binance-specific features
   - Spot trading
   - Futures trading (if needed)
   - Staking (if needed)

### Task 4.3: Integration & Testing

1. Integrate with `NormalizationService`
2. Test with Binance testnet
3. Validate crypto transaction handling
4. Create comprehensive tests

**Success Criteria**:
- ✓ API key authentication works
- ✓ Spot account data retrieval
- ✓ Transaction normalization correct
- ✓ Tests pass

---

## Technical Decisions

### Currency Conversion API
**Decision**: Use Fixer.io for fiat + CoinGecko for crypto
**Rationale**: 
- Fixer.io config already exists
- CoinGecko has free tier with good crypto coverage
- Separate providers allow independent fallbacks

### MCC Data Source
**Decision**: ISO 18245 standard + custom crypto mapping
**Rationale**:
- ISO standard provides comprehensive coverage
- Crypto needs custom logic (no standard MCCs)
- Can be stored in code or database

### Token Storage
**Decision**: Database storage with encryption
**Rationale**:
- Secure storage required for OAuth tokens
- Enables token refresh and management
- Audit trail for compliance

### Provider Selection Strategy
**Decision**: Provider field in PlaidItem table
**Rationale**:
- Already implemented in codebase
- Allows multiple providers per user
- Clean separation of concerns

---

## Risk Mitigation

### High Priority Risks
1. **OAuth Security**: Implement PKCE, validate state parameter, secure token storage
2. **API Rate Limits**: Implement rate limiting, request queuing, backoff strategies
3. **Data Consistency**: Transaction deduplication, idempotent operations
4. **Breaking Changes**: Comprehensive testing, feature flags, gradual rollout

### Medium Priority Risks
1. **Currency Conversion Accuracy**: Multiple data sources, validation, audit logging
2. **Performance**: Caching, async processing, database indexing
3. **Error Handling**: Graceful degradation, user-friendly messages, retry logic

---

## Success Metrics

### Phase 1 (Yapily)
- OAuth flow success rate: >99%
- API call success rate: >95%
- Data normalization accuracy: >98%
- Response time: <2s for account sync

### Phase 2 (NormalizationService)
- Currency conversion accuracy: ±0.1%
- MCC mapping coverage: >95%
- Conversion performance: <50ms
- Cache hit rate: >80%

### Phase 3 (Coinbase)
- Authentication success rate: >99%
- Transaction categorization accuracy: >90%
- Crypto conversion accuracy: ±0.5%

### Phase 4 (Binance)
- API integration success rate: >95%
- Data normalization accuracy: >95%
- Performance maintained: <3s for full sync

---

## Next Steps

1. ✅ Planning complete (Sequential Thinking)
2. ✅ API documentation gathered (Context7)
3. ✅ Codebase analyzed (Code Search)
4. ⏭️ Begin Phase 1: Yapily OAuth implementation
5. ⏭️ Create detailed task breakdown for Phase 1
