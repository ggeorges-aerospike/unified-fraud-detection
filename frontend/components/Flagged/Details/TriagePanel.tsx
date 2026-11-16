'use client'

import { useState } from 'react'
import useSWR from 'swr'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { ChevronDown, Loader2, RefreshCw, Scale } from 'lucide-react'

interface NoulAnswer { type: 'noul'; noul: number }
interface ChoiceAnswer { type: 'choice'; choice: string; confidence: number; probabilities: Record<string, number> }
type Answer = NoulAnswer | ChoiceAnswer

interface TriageDecision {
    outcome: 'auto_clear' | 'escalate' | 'escalate_urgent'
    rule: string
    explanation: string
    pattern?: string
    readings?: Record<string, 'yes' | 'no' | 'unsettled'>
    thresholds?: Record<string, [number, number]>
    model: string
    mode: string
    decided_at: string
    model_ms: number
    read_ms: number
    keys_read: number
    error?: string
    risk_score: number
    facts?: { device_confirmed_fraud: boolean; flagged_recipients: number; within_bank_norms?: boolean | null; norms_detail?: string }
    answers?: Record<string, Answer>
    request?: { model: string; state: unknown; questions: Record<string, { type: string; instructions: string }> }
}

const OUTCOME = {
    auto_clear: { label: 'Auto-cleared', cls: 'bg-teal-100 text-teal-800 border-teal-200' },
    escalate: { label: 'Escalated to agents', cls: 'bg-amber-100 text-amber-800 border-amber-200' },
    escalate_urgent: { label: 'Escalated · urgent', cls: 'bg-red-100 text-red-800 border-red-200' },
} as const

const READING = {
    yes: 'text-slate-900',
    no: 'text-slate-900',
    unsettled: 'text-amber-700',
} as const

const pretty = (s: string) => s.replace(/_/g, ' ')

export default function TriagePanel({ userId }: { userId: string }) {
    const { data, mutate, isLoading } = useSWR<{ decisions: TriageDecision[] }>(userId ? `/api/triage/${userId}` : null)
    const [running, setRunning] = useState(false)
    const d = data?.decisions?.[0]

    const rerun = async () => {
        setRunning(true)
        try {
            await fetch(`/api/triage/${userId}`, { method: 'POST' })
        } finally {
            setRunning(false)
            mutate()
        }
    }

    return (
        <Card className="bg-white border-slate-200 shadow-sm">
            <CardHeader className="pb-3">
                <div className="flex items-start justify-between gap-2">
                    <div>
                        <CardTitle className="flex items-center gap-2 text-slate-900">
                            <Scale className="h-5 w-5 text-teal-600" />
                            Decision-model triage
                        </CardTitle>
                        <CardDescription className="text-slate-500">
                            The model answers narrow questions; the rule that decides is in code.
                        </CardDescription>
                    </div>
                    <Button variant="outline" size="icon" className="h-8 w-8 border-slate-300" onClick={rerun} disabled={running} title="Re-run triage">
                        {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
                    </Button>
                </div>
            </CardHeader>
            <CardContent className="space-y-4">
                {isLoading ? (
                    <p className="text-sm text-slate-500">Loading…</p>
                ) : !d ? (
                    <p className="text-sm text-slate-500">Not triaged yet.</p>
                ) : (
                    <>
                        <div>
                            <span className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold ${OUTCOME[d.outcome]?.cls ?? ''}`}>
                                {OUTCOME[d.outcome]?.label ?? d.outcome}
                            </span>
                            {d.pattern && (
                                <span className="ml-2 text-xs text-slate-600">pattern: <b>{pretty(d.pattern)}</b></span>
                            )}
                            <p className="mt-2 text-sm text-slate-700">{d.explanation}</p>
                            <p className="mt-1 text-xs text-slate-500">Rule: <code>{d.rule}</code></p>
                            {d.error && <p className="mt-1 text-xs text-red-600">Model error: {d.error}</p>}
                        </div>

                        {d.answers && Object.keys(d.answers).length > 0 && (
                            <div className="space-y-2.5">
                                {Object.entries(d.answers).map(([name, a]) => {
                                    const question = d.request?.questions?.[name]?.instructions ?? pretty(name)
                                    if (a.type === 'noul') {
                                        const reading = d.readings?.[name] ?? 'unsettled'
                                        const [noAt, yesAt] = d.thresholds?.[name] ?? [0.3, 0.7]
                                        return (
                                            <div key={name}>
                                                <div className="flex items-baseline justify-between gap-3 text-xs">
                                                    <span className="text-slate-600">{question.replace(/`/g, '')}</span>
                                                    <span className={`shrink-0 font-semibold tabular-nums ${READING[reading]}`}>
                                                        {reading} · {a.noul.toFixed(2)}
                                                    </span>
                                                </div>
                                                <div className="relative mt-1 h-1.5 rounded-full bg-slate-100">
                                                    {/* the model's 'unsettled' band between its no / yes thresholds */}
                                                    <div className="absolute inset-y-0 bg-amber-100" style={{ left: `${noAt * 100}%`, right: `${(1 - yesAt) * 100}%` }} />
                                                    <div className="absolute inset-y-0 w-1 -ml-0.5 rounded-full bg-teal-700" style={{ left: `${a.noul * 100}%` }} />
                                                </div>
                                            </div>
                                        )
                                    }
                                    const probs = Object.entries(a.probabilities).sort((x, y) => y[1] - x[1])
                                    return (
                                        <div key={name} className="text-xs">
                                            <div className="text-slate-600">{question.replace(/`/g, '')}</div>
                                            <div className="mt-1 flex flex-wrap gap-1.5">
                                                {probs.map(([opt, pr]) => (
                                                    <span key={opt} className={`rounded border px-1.5 py-0.5 tabular-nums ${opt === a.choice ? 'border-teal-300 bg-teal-50 text-teal-900' : 'border-slate-200 text-slate-500'}`}>
                                                        {pretty(opt)} {pr.toFixed(2)}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    )
                                })}
                            </div>
                        )}

                        <div className="grid grid-cols-2 gap-2 text-xs">
                            <div className="rounded-md bg-slate-50 p-2">
                                <div className="text-slate-500">Facts from code</div>
                                <div className="mt-0.5 text-slate-800">
                                    Device tied to confirmed fraud: <b>{d.facts?.device_confirmed_fraud ? 'yes' : 'no'}</b>
                                </div>
                                <div className="text-slate-800">Recipients already decided as fraud: <b>{d.facts?.flagged_recipients ?? 0}</b></div>
                                <div className="text-slate-800">
                                    Within bank norms: <b>{d.facts?.within_bank_norms ? 'yes' : 'no'}</b>
                                    {d.facts?.norms_detail ? <span className="text-slate-500"> ({d.facts.norms_detail})</span> : null}
                                </div>
                            </div>
                            <div className="rounded-md bg-slate-50 p-2">
                                <div className="text-slate-500">Cost of this decision</div>
                                <div className="mt-0.5 text-slate-800 tabular-nums">
                                    Aerospike read <b>{d.read_ms.toFixed(1)} ms</b> · {d.keys_read} keys
                                </div>
                                <div className="text-slate-800 tabular-nums">
                                    {d.model} <b>{Math.round(d.model_ms)} ms</b>{d.mode === 'mock' ? ' (mock)' : ''}
                                </div>
                            </div>
                        </div>

                        {d.request && (
                            <Collapsible>
                                <CollapsibleTrigger className="flex items-center gap-1 text-xs font-medium text-slate-600 hover:text-slate-900">
                                    <ChevronDown className="h-3.5 w-3.5" />
                                    Exact request sent to the model (stored in Aerospike)
                                </CollapsibleTrigger>
                                <CollapsibleContent>
                                    <pre className="mt-2 max-h-80 overflow-auto rounded-md bg-slate-900 p-3 text-[11px] leading-relaxed text-slate-100">
                                        {JSON.stringify(d.request, null, 2)}
                                    </pre>
                                </CollapsibleContent>
                            </Collapsible>
                        )}
                        <p className="text-[11px] text-slate-400">Decided {new Date(d.decided_at).toLocaleString()}</p>
                    </>
                )}
            </CardContent>
        </Card>
    )
}
