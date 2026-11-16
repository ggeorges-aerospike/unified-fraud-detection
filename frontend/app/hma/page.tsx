'use client'

import { useMemo, useState } from 'react'
import useSWR from 'swr'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Cpu, Database, HardDrive, Info, Layers, RefreshCw } from 'lucide-react'

interface Tier {
    id: string
    label: string
    sets: string[]
    objects: number
    index_bytes: number
    data_bytes: number
    sindexes: number
    sindex_bytes: number
}

interface SIndex {
    name: string
    set: string | null
    bin: string | null
    type: string | null
    entries: number
    used_bytes: number
}

interface HmaStats {
    namespace: string
    nodes: number
    build: string
    edition: string
    records: number
    objects_all_copies: number
    ram: {
        primary_index_bytes: number
        secondary_index_bytes: number
        set_index_bytes: number
        total_bytes: number
        secondary_index_count: number
        secondary_index_uniform_allocation: number | null
    }
    ssd: {
        data_bytes: number
        capacity_bytes: number
        used_pct: number | null
        avail_pct: number | null
        devices: string[]
        kind: string
    }
    per_record: { index_bytes: number | null; data_bytes: number | null }
    config: {
        index_type: string | null
        sindex_type: string | null
        storage_engine: string | null
        replication_factor: number | null
        compression: string | null
        read_page_cache: string | null
        data_compression_ratio: string | null
    }
    tiers: Tier[]
    sindexes: SIndex[]
}

// Series colors (categorical slots 1-2), validated for both card surfaces with the dataviz
// validator: light #2a78d6 / #eb6834 on #ffffff, dark #3987e5 / #d95926 on #020817.
const VIZ_CSS = `
.hma-viz { --ram: #2a78d6; --ssd: #eb6834; }
[data-theme="dark"] .hma-viz, .dark .hma-viz { --ram: #3987e5; --ssd: #d95926; }
.hma-hatch { background-image: repeating-linear-gradient(45deg, var(--ram) 0 2px, transparent 2px 6px); }
`

const UNITS = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']
function fmtBytes(n: number | null | undefined): string {
    if (n == null || !isFinite(n)) return '–'
    let v = n
    let i = 0
    while (v >= 1024 && i < UNITS.length - 1) {
        v /= 1024
        i++
    }
    return `${v >= 100 || i === 0 ? Math.round(v) : v.toFixed(1)} ${UNITS[i]}`
}
const fmtNum = (n: number) => n.toLocaleString()
const fmtCount = (n: number) =>
    n >= 1e9 ? `${n / 1e9} billion` : n >= 1e6 ? `${n / 1e6} million` : fmtNum(n)

/** One horizontal stacked bar: rounded ends, 2px surface gaps, per-segment hover tooltip. */
function StackedBar({ segments }: { segments: { key: string; value: number; label: string; className: string; style?: React.CSSProperties }[] }) {
    const total = segments.reduce((s, x) => s + x.value, 0) || 1
    return (
        <div className="flex h-7 w-full gap-[2px]">
            {segments.filter((s) => s.value > 0).map((s, i, arr) => (
                <div
                    key={s.key}
                    className={`group relative h-full ${s.className} ${i === 0 ? 'rounded-l' : ''} ${i === arr.length - 1 ? 'rounded-r' : ''}`}
                    style={{ width: `${Math.max(0.6, (s.value / total) * 100)}%`, ...s.style }}
                    aria-label={`${s.label}: ${fmtBytes(s.value)}`}
                >
                    <div className="pointer-events-none absolute bottom-full left-1/2 z-10 mb-2 hidden -translate-x-1/2 whitespace-nowrap rounded-md border bg-popover px-2.5 py-1.5 text-xs text-popover-foreground shadow-md group-hover:block">
                        <div className="font-medium">{s.label}</div>
                        <div className="tabular-nums">{fmtBytes(s.value)} · {((s.value / total) * 100).toFixed(1)}%</div>
                    </div>
                </div>
            ))}
        </div>
    )
}

function LegendItem({ swatch, label }: { swatch: React.ReactNode; label: string }) {
    return (
        <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
            {swatch}
            {label}
        </span>
    )
}
const RamSwatch = () => <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: 'var(--ram)' }} />
const RamHatchSwatch = () => <span className="hma-hatch inline-block h-2.5 w-2.5 rounded-sm border" style={{ borderColor: 'var(--ram)' }} />
const SsdSwatch = () => <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: 'var(--ssd)' }} />

function Stat({ icon: Icon, title, value, sub }: { icon: React.ElementType; title: string; value: string; sub: React.ReactNode }) {
    return (
        <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">{title}</CardTitle>
                <Icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
                <div className="text-2xl font-bold tabular-nums">{value}</div>
                <div className="mt-1 text-xs text-muted-foreground">{sub}</div>
            </CardContent>
        </Card>
    )
}

const PRESETS = [1e8, 1e9, 1e10]

export default function HmaPage() {
    const { data, isLoading, error, mutate } = useSWR<HmaStats>('/api/storage/hma', { refreshInterval: 5000 })
    const [projRecords, setProjRecords] = useState(1e9)
    const [rf, setRf] = useState(2)

    const projection = useMemo(() => {
        if (!data?.per_record.data_bytes) return null
        const index = projRecords * 64 * rf
        const ssd = projRecords * data.per_record.data_bytes * rf
        return { index, ssd, allRam: index + ssd, ratio: (index + ssd) / index }
    }, [data, projRecords, rf])

    if (isLoading || (!data && !error)) {
        return (
            <div className="space-y-6">
                <Skeleton className="h-10 w-96" />
                <div className="grid gap-4 md:grid-cols-3">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-28" />)}</div>
                <Skeleton className="h-64" />
            </div>
        )
    }
    if (error || !data) {
        return <p className="text-destructive">Could not read Aerospike statistics. Is the backend connected to Aerospike?</p>
    }

    const { ram, ssd, per_record: pr } = data
    const fileBacked = ssd.kind === 'file'
    const ssdShareOfRecord = pr.index_bytes && pr.data_bytes ? (pr.data_bytes / (pr.index_bytes + pr.data_bytes)) * 100 : null

    return (
        <div className="hma-viz space-y-6">
            <style>{VIZ_CSS}</style>

            <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">Hybrid Memory Architecture</h1>
                    <p className="text-muted-foreground">
                        Each record&apos;s index entry lives in RAM; the record itself lives on flash. Read live from the cluster every 5 seconds.
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground">
                        {data.edition} {data.build} · namespace <code>{data.namespace}</code> · {data.nodes} node{data.nodes === 1 ? '' : 's'}
                    </p>
                </div>
                <Button variant="outline" onClick={() => mutate()}>
                    <RefreshCw className="mr-2 h-4 w-4" />
                    Refresh
                </Button>
            </div>

            {fileBacked && (
                <div className="flex items-start gap-2 rounded-lg border px-4 py-3 text-sm text-muted-foreground">
                    <Info className="mt-0.5 h-4 w-4 shrink-0" />
                    <span>
                        Local demo: the &quot;SSD&quot; here is a {fmtBytes(ssd.capacity_bytes)} file (<code>{ssd.devices.join(', ')}</code>) on
                        Docker&apos;s virtual disk. On AWS NVMe instances the same namespace writes to raw NVMe devices; the RAM side does not change.
                    </span>
                </div>
            )}

            {/* Headline tiles */}
            <div className="grid gap-4 md:grid-cols-3">
                <Stat icon={Database} title="Records" value={fmtNum(data.records)}
                      sub={<>Graph, predictive data, decision log and agent state in one namespace</>} />
                <Stat icon={Cpu} title="In RAM" value={fmtBytes(ram.total_bytes)}
                      sub={<>Primary index {fmtBytes(ram.primary_index_bytes)} · secondary indexes {fmtBytes(ram.secondary_index_bytes)}</>} />
                <Stat icon={HardDrive} title="On SSD" value={fmtBytes(ssd.data_bytes)}
                      sub={<>{ssd.used_pct ?? '–'}% of {fmtBytes(ssd.capacity_bytes)} {ssd.kind} capacity used</>} />
            </div>

            {/* Per record */}
            <Card>
                <CardHeader>
                    <CardTitle>One record, split across memory and flash</CardTitle>
                    <CardDescription>
                        Measured averages across all {fmtNum(data.objects_all_copies)} stored objects.
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                    <StackedBar segments={[
                        { key: 'idx', value: pr.index_bytes ?? 0, label: 'Primary-index entry (RAM)', className: '', style: { background: 'var(--ram)' } },
                        { key: 'data', value: pr.data_bytes ?? 0, label: 'Record data (SSD)', className: '', style: { background: 'var(--ssd)' } },
                    ]} />
                    <div className="flex flex-wrap items-center gap-x-6 gap-y-1 text-sm">
                        <span><b className="tabular-nums">{pr.index_bytes ?? '–'} bytes</b> <span className="text-muted-foreground">in RAM</span></span>
                        <span><b className="tabular-nums">{pr.data_bytes ?? '–'} bytes</b> <span className="text-muted-foreground">on SSD</span></span>
                        {ssdShareOfRecord != null && (
                            <span className="text-muted-foreground"><b className="text-foreground">{ssdShareOfRecord.toFixed(0)}%</b> of every record stays out of RAM</span>
                        )}
                        <span className="ml-auto flex gap-3"><LegendItem swatch={<RamSwatch />} label="RAM" /><LegendItem swatch={<SsdSwatch />} label="SSD" /></span>
                    </div>
                </CardContent>
            </Card>

            {/* Where the bytes live */}
            <Card>
                <CardHeader>
                    <CardTitle>Where the bytes live, right now</CardTitle>
                    <CardDescription>Everything the cluster holds for this namespace, by location.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                    <StackedBar segments={[
                        { key: 'pi', value: ram.primary_index_bytes, label: 'Primary index (RAM)', className: '', style: { background: 'var(--ram)' } },
                        { key: 'si', value: ram.secondary_index_bytes, label: 'Secondary indexes (RAM)', className: 'hma-hatch border', style: { borderColor: 'var(--ram)' } },
                        { key: 'ssd', value: ssd.data_bytes, label: 'Record data (SSD)', className: '', style: { background: 'var(--ssd)' } },
                    ]} />
                    <div className="flex flex-wrap gap-x-5 gap-y-1">
                        <LegendItem swatch={<RamSwatch />} label={`Primary index · ${fmtBytes(ram.primary_index_bytes)}`} />
                        <LegendItem swatch={<RamHatchSwatch />} label={`Secondary indexes · ${fmtBytes(ram.secondary_index_bytes)}`} />
                        <LegendItem swatch={<SsdSwatch />} label={`Record data · ${fmtBytes(ssd.data_bytes)}`} />
                    </div>
                    {ram.secondary_index_uniform_allocation != null && ram.secondary_index_count > 0 && (
                        <p className="text-xs text-muted-foreground">
                            Each of the {ram.secondary_index_count} secondary indexes currently holds a fixed {fmtBytes(ram.secondary_index_uniform_allocation)} reservation
                            regardless of how many entries it has, so at this small scale they dominate RAM. The primary index is the part that grows with every record.
                        </p>
                    )}
                </CardContent>
            </Card>

            {/* By tier */}
            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2"><Layers className="h-5 w-5" />Every tier of the funnel, one cluster</CardTitle>
                    <CardDescription>Sets grouped by the tier that owns them. Index RAM is 64 bytes × records.</CardDescription>
                </CardHeader>
                <CardContent className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b text-left text-xs text-muted-foreground">
                                <th className="py-2 pr-4 font-medium">Tier</th>
                                <th className="py-2 pr-4 text-right font-medium">Records</th>
                                <th className="py-2 pr-4 text-right font-medium">Index in RAM</th>
                                <th className="py-2 pr-4 text-right font-medium">Data on SSD</th>
                                <th className="py-2 pr-4 text-right font-medium">Secondary indexes</th>
                                <th className="py-2 font-medium">Sets</th>
                            </tr>
                        </thead>
                        <tbody>
                            {data.tiers.map((t) => (
                                <tr key={t.id} className="border-b last:border-0 align-top">
                                    <td className="py-2 pr-4 font-medium">{t.label}</td>
                                    <td className="py-2 pr-4 text-right tabular-nums">{fmtNum(t.objects)}</td>
                                    <td className="py-2 pr-4 text-right tabular-nums">{fmtBytes(t.index_bytes)}</td>
                                    <td className="py-2 pr-4 text-right tabular-nums">{fmtBytes(t.data_bytes)}</td>
                                    <td className="py-2 pr-4 text-right tabular-nums">{t.sindexes ? `${t.sindexes} · ${fmtBytes(t.sindex_bytes)}` : '–'}</td>
                                    <td className="py-2 text-xs text-muted-foreground">{t.sets.slice(0, 8).join(', ')}{t.sets.length > 8 ? ` +${t.sets.length - 8} more` : ''}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </CardContent>
            </Card>

            {/* Projection */}
            <Card>
                <CardHeader>
                    <CardTitle>At production scale</CardTitle>
                    <CardDescription>
                        Projected from this cluster&apos;s measured {pr.data_bytes ?? '–'} bytes of data per record and the fixed 64-byte index entry.
                        Secondary indexes are left out.
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-5">
                    <div className="flex flex-wrap items-center gap-2">
                        <span className="text-xs text-muted-foreground">Records</span>
                        {PRESETS.map((n) => (
                            <Button key={n} size="sm" variant={projRecords === n ? 'default' : 'outline'} onClick={() => setProjRecords(n)}>
                                {fmtCount(n)}
                            </Button>
                        ))}
                        <span className="ml-4 text-xs text-muted-foreground">Replication factor</span>
                        {[1, 2, 3].map((n) => (
                            <Button key={n} size="sm" variant={rf === n ? 'default' : 'outline'} onClick={() => setRf(n)}>{n}</Button>
                        ))}
                    </div>
                    {projection && (
                        <>
                            <div className="space-y-3">
                                <div>
                                    <div className="mb-1 flex justify-between text-sm">
                                        <span className="font-medium">Aerospike HMA: RAM</span>
                                        <span className="tabular-nums">{fmtBytes(projection.index)}</span>
                                    </div>
                                    <div className="h-4 w-full rounded bg-muted">
                                        <div className="h-4 rounded" style={{ width: `${(projection.index / projection.allRam) * 100}%`, background: 'var(--ram)' }} />
                                    </div>
                                </div>
                                <div>
                                    <div className="mb-1 flex justify-between text-sm">
                                        <span className="font-medium">Aerospike HMA: SSD</span>
                                        <span className="tabular-nums">{fmtBytes(projection.ssd)}</span>
                                    </div>
                                    <div className="h-4 w-full rounded bg-muted">
                                        <div className="h-4 rounded" style={{ width: `${(projection.ssd / projection.allRam) * 100}%`, background: 'var(--ssd)' }} />
                                    </div>
                                </div>
                                <div>
                                    <div className="mb-1 flex justify-between text-sm">
                                        <span className="font-medium">All-in-RAM store: RAM, at minimum</span>
                                        <span className="tabular-nums">{fmtBytes(projection.allRam)}</span>
                                    </div>
                                    <div className="flex h-4 w-full gap-[2px]">
                                        <div className="h-4 rounded-l" style={{ width: `${(projection.index / projection.allRam) * 100}%`, background: 'var(--ram)' }} />
                                        <div className="h-4 flex-1 rounded-r hma-hatch border" style={{ borderColor: 'var(--ram)' }} />
                                    </div>
                                </div>
                            </div>
                            <p className="text-sm">
                                <b>{fmtCount(projRecords)} records</b> at replication factor {rf}: Aerospike needs <b>{fmtBytes(projection.index)}</b> of RAM for the index
                                and keeps <b>{fmtBytes(projection.ssd)}</b> on NVMe. Holding the same data in RAM takes at least <b>{fmtBytes(projection.allRam)}</b>,
                                about <b>{projection.ratio.toFixed(1)}×</b> the RAM.
                            </p>
                            <p className="text-xs text-muted-foreground">
                                &quot;At minimum&quot; counts only the raw record bytes plus a 64-byte key entry; real in-memory stores add their own per-key overhead on top.
                            </p>
                        </>
                    )}
                </CardContent>
            </Card>

            {/* Configuration facts */}
            <Card>
                <CardHeader>
                    <CardTitle>How this namespace is configured</CardTitle>
                    <CardDescription>Read from the server.</CardDescription>
                </CardHeader>
                <CardContent>
                    <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm md:grid-cols-4">
                        {[
                            ['Primary index', `${data.config.index_type ?? '–'}${data.config.index_type === 'shmem' ? ' (RAM)' : ''}`],
                            ['Secondary indexes', `${data.config.sindex_type ?? '–'}${data.config.sindex_type === 'shmem' ? ' (RAM)' : ''}`],
                            ['Storage engine', `${data.config.storage_engine ?? '–'} · ${ssd.kind}`],
                            ['Devices', ssd.devices.join(', ') || '–'],
                            ['Replication factor', String(data.config.replication_factor ?? '–')],
                            ['Compression', `${data.config.compression ?? '–'} (ratio ${data.config.data_compression_ratio ?? '–'})`],
                            ['Read page cache', data.config.read_page_cache ?? '–'],
                            ['Capacity available', ssd.avail_pct != null ? `${ssd.avail_pct}%` : '–'],
                        ].map(([k, v]) => (
                            <div key={k}>
                                <dt className="text-xs text-muted-foreground">{k}</dt>
                                <dd className="break-all font-medium">{v}</dd>
                            </div>
                        ))}
                    </dl>
                </CardContent>
            </Card>
        </div>
    )
}
