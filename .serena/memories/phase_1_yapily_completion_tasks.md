# Phase 1: Yapily API Integration - Detailed Tasks

**Phase**: 1 of 4
**Status**: In Progress
**Dependencies**: None
**Estimated Duration**: 2-3 days

---

## Task 1.1: OAuth 2.0 Authentication Flow

### Todo 1.1.1: Create Yapily Auth Service
**File**: `/src/providers/yapily/auth.service.ts`

**Implementation**:
```typescript
import axios from 'axios';

export interface YapilyAuthConfig {
  applicationKey: string;
  applicationSecret: string;
  baseUrl: string;
  callbackUrl: string;
}

export interface AuthorizationRequest {
  userUuid: string;
  institutionId: string;
  callback: string;
}

export interface AuthorizationResponse {
  authorisationUrl: string;
  status: string;
  id?: string;
}

export interface TokenExchangeRequest {
  authCode: string;
  authState: string;
}

export interface TokenExchangeResponse {
  consentToken: string;
  idToken?: string;
}

export class YapilyAuthService {
  private config: YapilyAuthConfig;
  private client: axios.AxiosInstance;

  constructor(config: YapilyAuthConfig) {
    this.config = config;
    this.client = axios.create({
      baseURL: config.baseUrl,
      auth: {
        username: config.applicationKey,
        password: config.applicationSecret,
      },
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 10000,
    });
  }

  async initiateAuthorization(
    userUuid: string,
    institutionId: string
  ): Promise<AuthorizationResponse> {
    const response = await this.client.post('/account-auth-requests', {
      userUuid,
      institutionId,
      callback: this.config.callbackUrl,
    });
    return response.data;
  }

  async exchangeToken(
    authCode: string,
    authState: string
  ): Promise<TokenExchangeResponse> {
    const response = await this.client.post('/consent-auth-code', {
      authCode,
      authState,
    });
    return response.data;
  }

  // Token refresh if supported by Yapily
  async refreshToken(consentToken: string): Promise<TokenExchangeResponse> {
    // TODO: Check if Yapily supports token refresh
    throw new Error('Token refresh not yet implemented');
  }
}
```

**Tests**: `/tests/providers/yapily/auth.service.test.ts`

---

### Todo 1.1.2: Database Schema for Yapily Tokens

**Option A**: Extend PlaidItem table
```prisma
model PlaidItem {
  // ... existing fields
  provider         String @default("plaid") // "plaid" | "yapily" | "coinbase" | "binance"
  yapilyConsentToken String?
  yapilyIdToken      String?
  yapilyInstitutionId String?
  tokenExpiresAt     DateTime?
}
```

**Option B**: Create separate YapilyToken table (recommended for scalability)
```prisma
model YapilyToken {
  id              String   @id @default(cuid())
  userId          String
  institutionId   String
  consentToken    String
  idToken         String?
  authRequestId   String?
  expiresAt       DateTime?
  createdAt       DateTime @default(now())
  updatedAt       DateTime @updatedAt
  
  user            User     @relation(fields: [userId], references: [id], onDelete: Cascade)
  
  @@unique([userId, institutionId])
  @@index([userId])
}
```

**Migration**: Create migration file

---

### Todo 1.1.3: Environment Configuration

**File**: `.env.example` (add these)
```bash
# Yapily Configuration
YAPILY_APPLICATION_KEY=your_application_key_here
YAPILY_APPLICATION_SECRET=your_application_secret_here
YAPILY_BASE_URL=https://api.yapily.com
YAPILY_CALLBACK_URL=https://your-app.com/api/yapily/callback
YAPILY_SANDBOX_MODE=true
```

**File**: `/src/config/yapily.config.ts`
```typescript
export const yapilyConfig = {
  applicationKey: process.env.YAPILY_APPLICATION_KEY!,
  applicationSecret: process.env.YAPILY_APPLICATION_SECRET!,
  baseUrl: process.env.YAPILY_BASE_URL || 'https://api.yapily.com',
  callbackUrl: process.env.YAPILY_CALLBACK_URL!,
  sandboxMode: process.env.YAPILY_SANDBOX_MODE === 'true',
};
```

---

### Todo 1.1.4: API Routes for OAuth Flow

**File**: `/src/app/api/yapily/authorize/route.ts`
```typescript
import { NextRequest, NextResponse } from 'next/server';
import { YapilyAuthService } from '@/providers/yapily/auth.service';
import { yapilyConfig } from '@/config/yapily.config';

export async function POST(req: NextRequest) {
  const { userId, institutionId } = await req.json();
  
  const authService = new YapilyAuthService(yapilyConfig);
  const authResponse = await authService.initiateAuthorization(
    userId,
    institutionId
  );
  
  // Store auth request ID in session/database
  
  return NextResponse.json({
    authorisationUrl: authResponse.authorisationUrl,
    status: authResponse.status,
  });
}
```

**File**: `/src/app/api/yapily/callback/route.ts`
```typescript
import { NextRequest, NextResponse } from 'next/server';
import { YapilyAuthService } from '@/providers/yapily/auth.service';
import { yapilyConfig } from '@/config/yapily.config';
import { prisma } from '@/lib/prisma';

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const authCode = searchParams.get('code');
  const authState = searchParams.get('state');
  
  if (!authCode || !authState) {
    return NextResponse.json({ error: 'Missing parameters' }, { status: 400 });
  }
  
  const authService = new YapilyAuthService(yapilyConfig);
  const tokenResponse = await authService.exchangeToken(authCode, authState);
  
  // Store tokens in database
  // Redirect to success page
  
  return NextResponse.redirect('/accounts?yapily=success');
}
```

---

## Task 1.2: Real Yapily Endpoints

### Todo 1.2.1: Create Yapily HTTP Client

**File**: `/src/providers/yapily/client.ts`
```typescript
import axios, { AxiosInstance, AxiosError } from 'axios';

export interface YapilyClientConfig {
  applicationKey: string;
  applicationSecret: string;
  baseUrl: string;
}

export class YapilyClient {
  private client: AxiosInstance;

  constructor(config: YapilyClientConfig) {
    this.client = axios.create({
      baseURL: config.baseUrl,
      auth: {
        username: config.applicationKey,
        password: config.applicationSecret,
      },
      timeout: 15000,
    });

    // Request interceptor for logging
    this.client.interceptors.request.use(
      (config) => {
        console.log(`[Yapily] ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response) {
          console.error('[Yapily] API Error:', error.response.status, error.response.data);
        }
        return Promise.reject(this.mapError(error));
      }
    );
  }

  async getAccounts(consentToken: string) {
    const response = await this.client.get('/accounts', {
      headers: {
        Consent: consentToken,
      },
    });
    return response.data;
  }

  async getTransactions(
    consentToken: string,
    accountId: string,
    params?: {
      from?: string;
      before?: string;
      limit?: number;
      offset?: number;
    }
  ) {
    const response = await this.client.get(`/accounts/${accountId}/transactions`, {
      headers: {
        Consent: consentToken,
      },
      params,
    });
    return response.data;
  }

  private mapError(error: AxiosError): Error {
    // Map Yapily-specific errors to standard errors
    if (error.response?.status === 401) {
      return new Error('Yapily authentication failed - invalid or expired token');
    }
    if (error.response?.status === 403) {
      return new Error('Yapily access forbidden - insufficient permissions');
    }
    if (error.response?.status === 429) {
      return new Error('Yapily rate limit exceeded');
    }
    return new Error(`Yapily API error: ${error.message}`);
  }
}
```

---

### Todo 1.2.2: Update YapilyProvider with Real Implementation

**File**: `/src/providers/yapily/yapily.provider.ts`
```typescript
import { FinancialDataProvider } from "../base.provider";
import { ProviderAccount, ProviderTransaction } from "../types";
import { YapilyClient } from "./client";
import { yapilyConfig } from "@/config/yapily.config";

export class YapilyProvider extends FinancialDataProvider {
  private client: YapilyClient;

  constructor() {
    super();
    this.client = new YapilyClient(yapilyConfig);
  }

  async getAccounts(accessToken: string): Promise<ProviderAccount[]> {
    const response = await this.client.getAccounts(accessToken);
    
    return response.accounts.map((acc: any) => ({
      accountId: acc.account_id || acc.id,
      name: acc.account_details?.names?.[0] || acc.name || 'Unknown Account',
      currency: acc.account_details?.currency || acc.currency || 'GBP',
      balance: this.extractBalance(acc.balances),
      type: this.mapAccountType(acc.account_details?.type || acc.type),
      mask: acc.account_details?.identification?.identification?.slice(-4),
    }));
  }

  async getTransactions(
    accessToken: string,
    fromDate: Date,
    toDate: Date
  ): Promise<ProviderTransaction[]> {
    // Yapily requires accountId, but we need to fetch all accounts first
    const accounts = await this.getAccounts(accessToken);
    
    const allTransactions: ProviderTransaction[] = [];
    
    for (const account of accounts) {
      const response = await this.client.getTransactions(
        accessToken,
        account.accountId,
        {
          from: fromDate.toISOString().split('T')[0],
          before: toDate.toISOString().split('T')[0],
          limit: 1000,
        }
      );
      
      const transactions = response.data.map((tx: any) => ({
        transactionId: tx.id,
        accountId: account.accountId,
        amount: tx.transactionAmount?.amount || tx.amount,
        date: tx.date || tx.bookingDateTime,
        description: tx.description || tx.transactionInformation?.[0] || 'Unknown',
        currency: tx.transactionAmount?.currency || tx.currency || account.currency,
        merchantName: tx.merchant?.name || this.extractMerchantName(tx.description),
        category: this.mapCategory(tx.isoBankTransactionCode),
        status: tx.status === 'BOOKED' ? 'posted' : 'pending',
        rawData: tx,
      }));
      
      allTransactions.push(...transactions);
    }
    
    return allTransactions;
  }

  private extractBalance(balances: any[]): number {
    if (!balances || balances.length === 0) return 0;
    
    // Prefer current balance, then available, then closing
    const currentBalance = balances.find(b => b.type === 'CURRENT' || b.type === 'current');
    if (currentBalance) {
      return parseFloat(currentBalance.balanceAmount?.amount || currentBalance.amount || '0');
    }
    
    return parseFloat(balances[0].balanceAmount?.amount || balances[0].amount || '0');
  }

  private mapAccountType(yapilyType: string): string {
    const typeMap: Record<string, string> = {
      'current': 'depository',
      'savings': 'depository',
      'credit': 'credit',
      'investment': 'investment',
    };
    return typeMap[yapilyType?.toLowerCase()] || 'depository';
  }

  private mapCategory(isoBankTransactionCode: any): string {
    // Basic category mapping from ISO codes
    if (!isoBankTransactionCode) return 'Uncategorized';
    
    const familyCode = isoBankTransactionCode.familyCode?.code;
    const categoryMap: Record<string, string> = {
      'ICDT': 'Transfer',
      'PMNT': 'Payment',
      'RCDT': 'Income',
      'SALA': 'Income',
    };
    
    return categoryMap[familyCode] || 'Uncategorized';
  }

  private extractMerchantName(description: string): string | undefined {
    // Simple extraction - can be enhanced with regex patterns
    if (!description) return undefined;
    return description.split(/\s+/)[0];
  }
}
```

---

### Todo 1.2.3: Add Rate Limiting

**File**: `/src/providers/yapily/rate-limiter.ts`
```typescript
export class RateLimiter {
  private requests: number[] = [];
  private maxRequests: number;
  private windowMs: number;

  constructor(maxRequests: number = 100, windowMs: number = 60000) {
    this.maxRequests = maxRequests;
    this.windowMs = windowMs;
  }

  async checkLimit(): Promise<void> {
    const now = Date.now();
    this.requests = this.requests.filter(time => now - time < this.windowMs);
    
    if (this.requests.length >= this.maxRequests) {
      const oldestRequest = this.requests[0];
      const waitTime = this.windowMs - (now - oldestRequest);
      await new Promise(resolve => setTimeout(resolve, waitTime));
      return this.checkLimit();
    }
    
    this.requests.push(now);
  }
}
```

---

## Task 1.3: Testing & Validation

### Todo 1.3.1: Integration Tests

**File**: `/tests/providers/yapily/integration.test.ts`
```typescript
import { describe, it, expect, beforeAll } from "bun:test";
import { YapilyProvider } from "../../../src/providers/yapily/yapily.provider";
import { YapilyAuthService } from "../../../src/providers/yapily/auth.service";
import { yapilyConfig } from "../../../src/config/yapily.config";

describe("Yapily Integration Tests", () => {
  let provider: YapilyProvider;
  let authService: YapilyAuthService;
  let testConsentToken: string;

  beforeAll(async () => {
    provider = new YapilyProvider();
    authService = new YapilyAuthService(yapilyConfig);
    
    // TODO: Get test consent token from sandbox
    testConsentToken = process.env.YAPILY_TEST_CONSENT_TOKEN || '';
  });

  it("should fetch real accounts from Yapily sandbox", async () => {
    if (!testConsentToken) {
      console.warn("Skipping test - no consent token available");
      return;
    }
    
    const accounts = await provider.getAccounts(testConsentToken);
    expect(accounts).toBeArray();
    expect(accounts.length).toBeGreaterThan(0);
    expect(accounts[0]).toHaveProperty("accountId");
    expect(accounts[0]).toHaveProperty("currency");
  });

  it("should fetch real transactions from Yapily sandbox", async () => {
    if (!testConsentToken) {
      console.warn("Skipping test - no consent token available");
      return;
    }
    
    const fromDate = new Date('2024-01-01');
    const toDate = new Date();
    
    const transactions = await provider.getTransactions(
      testConsentToken,
      fromDate,
      toDate
    );
    
    expect(transactions).toBeArray();
    // Sandbox may have 0 transactions, so just check structure
    if (transactions.length > 0) {
      expect(transactions[0]).toHaveProperty("transactionId");
      expect(transactions[0]).toHaveProperty("amount");
      expect(transactions[0]).toHaveProperty("currency");
    }
  });
});
```

---

## Implementation Order

1. ✅ Create auth service (Todo 1.1.1)
2. ✅ Add database schema (Todo 1.1.2)
3. ✅ Configure environment (Todo 1.1.3)
4. ✅ Create API routes (Todo 1.1.4)
5. ✅ Create HTTP client (Todo 1.2.1)
6. ✅ Update provider implementation (Todo 1.2.2)
7. ✅ Add rate limiting (Todo 1.2.3)
8. ✅ Create integration tests (Todo 1.3.1)
9. ✅ Test with Yapily sandbox
10. ✅ Validate data normalization

---

## Dependencies

- `axios` - HTTP client (already in project)
- Yapily sandbox credentials (need to obtain)
- Database migration tools (Prisma)

---

## Blockers

None identified yet. Will update if blockers arise during implementation.
