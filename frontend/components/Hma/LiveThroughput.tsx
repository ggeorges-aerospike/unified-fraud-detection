'use client'

import { useEffect, useRef, useState } from 'react'
import useSWR from 'swr'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Activity } from 'lucide-react'

interface Throughput {
    writes_per_sec: number
    reads_per_sec: number
    write_over_ms: Record<string, number | null>
    read_over_ms: Record<string, number | null>
    nodes: { node: string; writes_per_sec: number; reads_per_sec: number }[]
    stream: { sets: string[]; records: number; index_bytes: number; data_bytes: number }
    replication_factor: number
}

// Categorical slot 3 (aqua), validated all-pairs with the page's RAM blue and SSD orange.
// Light-mode contrast is low, so every mark below carries a visible number.
const TPUT_CSS = `
.tput-viz { --tput: #1baf7a; }
[data-theme="dark"] .tput-viz, .dark .tput-viz { --tput: #199e70; }
`

const POLL_MS = 2000
const HISTORY = 90 // 3 minutes at 2 s

const fmtRate = (n: number) =>
    n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(n >= 1e5 ? 0 : 1)}k` : Math.round(n).toString()
const UNITS = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']
function fmtBytes(n: number): string {
    let v = n
    let i = 0
    while (v >= 1024 && i < UNITS.length - 1) {
        v /= 1024
        i++
    }
    return `${v >= 100 || i === 0 ? Math.round(v) : v.toFixed(1)} ${UNITS[i]}`
}
const pct = (v: number | null | undefined) => (v == null ? '–' : `${v.toFixed(2)}%`)

/** Writes/sec over the last few minutes: one column per sample, hover shows the value. */
function Sparkline({ samples, peak }: { samples: { t: number; v: number }[]; peak: number }) {
    const max = Math.max(peak, 1)
    const padded = [...Array(Math.max(0, HISTORY - samples.length)).fill(null), ...samples]
    return (
        <div className="flex h-24 w-full items-end gap-[2px]" role="img" aria-label="Writes per second over the last 3 minutes">
            {padded.map((s, i) =>
                s == null ? (
                    <div key={`e${i}`} className="h-full flex-1" />
                ) : (
                    <div key={s.t} className="group relative flex h-full flex-1 items-end">
                        <div
                            className="w-full rounded-t-[2px]"
                            style={{ height: `${Math.max(s.v > 0 ? 2 : 0, (s.v / max) * 100)}%`, background: 'var(--tput)' }}
                        />
                        <div className="pointer-events-none absolute bottom-full left-1/2 z-10 mb-1 hidden -translate-x-1/2 whitespace-nowrap rounded-md border bg-popover px-2 py-1 text-xs text-popover-foreground shadow-md group-hover:block">
                            <span className="tabular-nums font-medium">{fmtRate(s.v)} writes/s</span>
                            <span className="ml-1 text-muted-foreground">{new Date(s.t).toLocaleTimeString()}</span>
                        </div>
                    </div>
                ),
            )}
        </div>
    )
}

export default function LiveThroughput() {
    const { data, error } = useSWR<Throughput>('/api/storage/throughput', { refreshInterval: POLL_MS })
    const [samples, setSamples] = useState<{ t: number; v: number }[]>([])
    const peak = useRef(0)

    useEffect(() => {
        if (!data) return
        peak.current = Math.max(peak.current, data.writes_per_sec)
        setSamples((prev) => [...prev, { t: Date.now(), v: data.writes_per_sec }].slice(-HISTORY))
    }, [data])

    if (error) return null // the rest of the HMA page still renders
    if (!data) return null

    const nodeMax = Math.max(...data.nodes.map((n) => n.writes_per_sec), 1)
    const idle = data.writes_per_sec < 1

    return (
        <Card className="tput-viz">
            <style>{TPUT_CSS}</style>
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Activity className="h-5 w-5" />
                    Write throughput, live
                </CardTitle>
                <CardDescription>
                    Client writes per second across all {data.nodes.length} nodes, from the server&apos;s own statistics, every {POLL_MS / 1000} seconds.
                    {idle && ' The cluster is idle; start the load generator to see it move.'}
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
                <div className="grid gap-4 sm:grid-cols-4">
                    <div>
                        <div className="text-xs text-muted-foreground">Writes / sec</div>
                        <div className="text-3xl font-bold tabular-nums">{fmtRate(data.writes_per_sec)}</div>
                        <div className="text-xs text-muted-foreground">peak this session {fmtRate(peak.current)}</div>
                    </div>
                    <div>
                        <div className="text-xs text-muted-foreground">Writes slower than 1 ms</div>
                        <div className="text-3xl font-bold tabular-nums">{pct(data.write_over_ms['1'])}</div>
                        <div className="text-xs text-muted-foreground">over 8 ms {pct(data.write_over_ms['8'])} · over 64 ms {pct(data.write_over_ms['64'])}</div>
                    </div>
                    <div>
                        <div className="text-xs text-muted-foreground">Transactions held</div>
                        <div className="text-3xl font-bold tabular-nums">{data.stream.records.toLocaleString()}</div>
                        <div className="text-xs text-muted-foreground">in <code>{data.stream.sets.join(', ')}</code>, replication factor {data.replication_factor}</div>
                    </div>
                    <div>
                        <div className="text-xs text-muted-foreground">Their RAM vs NVMe</div>
                        <div className="text-3xl font-bold tabular-nums">{fmtBytes(data.stream.index_bytes)}</div>
                        <div className="text-xs text-muted-foreground">index in RAM · {fmtBytes(data.stream.data_bytes)} of data on NVMe</div>
                    </div>
                </div>

                <div>
                    <div className="mb-1 flex justify-between text-xs text-muted-foreground">
                        <span>Last 3 minutes</span>
                        <span className="tabular-nums">scale 0 – {fmtRate(Math.max(peak.current, 1))}/s</span>
                    </div>
                    <Sparkline samples={samples} peak={peak.current} />
                </div>

                <div className="space-y-2">
                    <div className="text-xs text-muted-foreground">Per node</div>
                    {data.nodes.map((n) => (
                        <div key={n.node} className="flex items-center gap-3 text-sm">
                            <span className="w-40 shrink-0 truncate font-mono text-xs text-muted-foreground" title={n.node}>{n.node}</span>
                            <div className="h-3 flex-1 rounded bg-muted">
                                <div className="h-3 rounded" style={{ width: `${(n.writes_per_sec / nodeMax) * 100}%`, background: 'var(--tput)' }} />
                            </div>
                            <span className="w-20 shrink-0 text-right tabular-nums">{fmtRate(n.writes_per_sec)}/s</span>
                        </div>
                    ))}
                </div>
            </CardContent>
        </Card>
    )
}
