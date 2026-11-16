'use client'

import { useState } from 'react'
import useSWR from 'swr'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Skeleton } from '@/components/ui/skeleton'
import { Activity, Bot, ChevronRight, Database, Gauge, Loader2, Scale, UserCheck } from 'lucide-react'

export interface TriageRun {
    requested: number
    decided: number
    wall_seconds: number
    decisions_per_second: number
    concurrency: number
    keys_read: number
    outcomes: { auto_clear: number; escalate: number; escalate_urgent: number }
    latency_ms: {
        model_p50: number | null
        model_p95: number | null
        state_read_p50: number | null
        state_read_p95: number | null
        write_p50: number | null
        write_p95: number | null
    }
}

interface Funnel {
    predictive: { evaluated: number | null; flagged: number; threshold: number; last_run: string | null; model?: string | null }
    decision_model: {
        enabled: boolean
        mode: string
        model: string | null
        provider: string | null
        triaged: number
        untriaged_pending: number
        auto_cleared: number
        escalated: number
        urgent: number
        last_run: TriageRun | null
    }
    agents: { investigated: number; awaiting: number; engine?: string; label?: string; model?: string | null }
    human: { decided: number }
}

const fmt = (n: number | null | undefined) => (n ?? 0).toLocaleString()
const ms = (n: number | null | undefined) => (n == null ? '–' : `${n < 10 ? n.toFixed(1) : Math.round(n)} ms`)

function Tier({
    step, title, icon: Icon, value, caption, detail, accent,
}: {
    step: number
    title: string
    icon: React.ElementType
    value: string
    caption: string
    detail?: React.ReactNode
    accent: string
}) {
    return (
        <div className="flex-1 min-w-0 rounded-lg border bg-card p-4">
            <div className="flex items-start gap-2 text-xs font-medium text-muted-foreground">
                <span className={`inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold text-white ${accent}`}>
                    {step}
                </span>
                <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <span className="min-w-0 leading-snug">{title}</span>
            </div>
            <div className="mt-2 text-3xl font-bold tabular-nums">{value}</div>
            <p className="text-xs text-muted-foreground">{caption}</p>
            {detail && <div className="mt-2 text-xs">{detail}</div>}
        </div>
    )
}

export default function TriageFunnel({ onChanged }: { onChanged?: () => void }) {
    const { data, isLoading, mutate } = useSWR<Funnel>('/api/triage/funnel', { refreshInterval: 10000 })
    const [threshold, setThreshold] = useState<string>('')
    const [busy, setBusy] = useState<null | 'detect' | 'triage'>(null)
    const [error, setError] = useState<string | null>(null)

    const refresh = () => {
        mutate()
        onChanged?.()
    }

    const runDetection = async () => {
        setBusy('detect')
        setError(null)
        try {
            if (threshold !== '') {
                const res = await fetch('/api/detection/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ risk_threshold: Number(threshold) }),
                })
                if (!res.ok) throw new Error(`Could not set threshold (${res.status})`)
            }
            const res = await fetch('/api/flagged-accounts/detect?skip_cooldown=true', { method: 'POST' })
            if (!res.ok) throw new Error(`Detection failed (${res.status})`)
            setThreshold('')
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e))
        } finally {
            setBusy(null)
            refresh()
        }
    }

    const runTriage = async () => {
        setBusy('triage')
        setError(null)
        try {
            const res = await fetch('/api/triage/run', { method: 'POST' })
            if (!res.ok) throw new Error(`Triage failed (${res.status})`)
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e))
        } finally {
            setBusy(null)
            refresh()
        }
    }

    if (isLoading || !data) {
        return (
            <Card>
                <CardHeader><Skeleton className="h-5 w-64" /></CardHeader>
                <CardContent><Skeleton className="h-28 w-full" /></CardContent>
            </Card>
        )
    }

    const p = data.predictive
    const dm = data.decision_model
    const run = dm.last_run
    const escalatedNormal = dm.escalated - dm.urgent

    return (
        <Card>
            <CardHeader className="pb-3">
                <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
                    <div>
                        <CardTitle>Fraud decision funnel</CardTitle>
                        <CardDescription>
                            Each tier handles fewer, harder cases. One Aerospike cluster serves every tier.
                        </CardDescription>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                        <label className="text-xs text-muted-foreground" htmlFor="funnel-threshold">
                            Predictive threshold
                        </label>
                        <Input
                            id="funnel-threshold"
                            type="number"
                            min={0}
                            max={100}
                            placeholder={String(p.threshold)}
                            value={threshold}
                            onChange={(e) => setThreshold(e.target.value)}
                            className="h-8 w-20"
                        />
                        <Button size="sm" variant="outline" onClick={runDetection} disabled={busy !== null}>
                            {busy === 'detect' ? <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" /> : <Activity className="mr-2 h-3.5 w-3.5" />}
                            Run detection
                        </Button>
                        {dm.untriaged_pending > 0 && dm.enabled && (
                            <Button size="sm" onClick={runTriage} disabled={busy !== null}>
                                {busy === 'triage' ? <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" /> : <Scale className="mr-2 h-3.5 w-3.5" />}
                                Triage {fmt(dm.untriaged_pending)} pending
                            </Button>
                        )}
                    </div>
                </div>
                {error && <p className="text-sm text-destructive">{error}</p>}
            </CardHeader>
            <CardContent className="space-y-3">
                <div className="flex flex-col items-stretch gap-2 md:flex-row">
                    <Tier
                        step={1}
                        title={p.model ? `Predictive model · ${p.model}` : 'Predictive model'}
                        icon={Gauge}
                        accent="bg-slate-500"
                        value={fmt(p.flagged)}
                        caption={`flagged of ${fmt(p.evaluated)} scored · threshold ${p.threshold}`}
                    />
                    <ChevronRight className="hidden h-5 w-5 shrink-0 self-center text-muted-foreground md:block" />
                    <Tier
                        step={2}
                        title={dm.provider ? `Decision model · ${dm.provider}` : 'Decision model'}
                        icon={Scale}
                        accent="bg-teal-600"
                        value={dm.enabled ? fmt(dm.escalated) : 'off'}
                        caption={dm.enabled ? 'escalated to the agents' : 'DECISION_MODEL=off'}
                        detail={
                            dm.enabled && (
                                <div className="flex flex-wrap gap-x-3 gap-y-1">
                                    <span className="font-medium text-teal-700 dark:text-teal-400">{fmt(dm.auto_cleared)} auto-cleared</span>
                                    <span className="text-red-600 dark:text-red-400">{fmt(dm.urgent)} urgent</span>
                                    <span className="text-muted-foreground">{fmt(escalatedNormal)} normal</span>
                                    {dm.model && (
                                        <span className="rounded bg-muted px-1.5 text-[10px] tracking-wide text-muted-foreground">{dm.model}</span>
                                    )}
                                </div>
                            )
                        }
                    />
                    <ChevronRight className="hidden h-5 w-5 shrink-0 self-center text-muted-foreground md:block" />
                    <Tier
                        step={3}
                        title={data.agents.label ? `AI agents · ${data.agents.label}` : 'AI agents'}
                        icon={Bot}
                        accent="bg-indigo-600"
                        value={fmt(data.agents.investigated)}
                        caption={`investigated · ${fmt(data.agents.awaiting)} awaiting`}
                        detail={
                            data.agents.model && (
                                <span className="rounded bg-muted px-1.5 text-[10px] tracking-wide text-muted-foreground">{data.agents.model}</span>
                            )
                        }
                    />
                    <ChevronRight className="hidden h-5 w-5 shrink-0 self-center text-muted-foreground md:block" />
                    <Tier
                        step={4}
                        title="Analyst"
                        icon={UserCheck}
                        accent="bg-amber-600"
                        value={fmt(data.human.decided)}
                        caption="final decisions"
                    />
                </div>

                {run && run.decided > 0 && (
                    <div className="flex flex-wrap items-center gap-x-5 gap-y-1 rounded-md bg-muted/60 px-3 py-2 text-xs text-muted-foreground">
                        <span className="flex items-center gap-1.5 font-medium text-foreground">
                            <Database className="h-3.5 w-3.5" />
                            Last triage run
                        </span>
                        <span>
                            <b className="text-foreground tabular-nums">{fmt(run.decided)}</b> decisions in {run.wall_seconds.toFixed(1)} s
                            {' '}(<b className="text-foreground tabular-nums">{fmt(run.decisions_per_second)}</b>/s, {run.concurrency} parallel)
                        </span>
                        <span>Aerospike state read p50 <b className="text-foreground tabular-nums">{ms(run.latency_ms.state_read_p50)}</b></span>
                        <span>decision write p50 <b className="text-foreground tabular-nums">{ms(run.latency_ms.write_p50)}</b></span>
                        <span>model p50 <b className="text-foreground tabular-nums">{ms(run.latency_ms.model_p50)}</b></span>
                        <span>{fmt(run.keys_read)} keys read</span>
                    </div>
                )}
            </CardContent>
        </Card>
    )
}
