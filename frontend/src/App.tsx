import { useEffect, useState } from 'react'
import './App.css'
import './Output.css'

function App() {
  const [message, setMessage] = useState<string>('Fetching a random insight...')
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const apiBase = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

  const fetchRandomInsight = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const response = await fetch(`${apiBase}/random`)
      if (!response.ok) {
        throw new Error(`Backend responded with ${response.status}`)
      }

      const data: { message?: string } = await response.json()
      setMessage(data.message ?? 'Backend did not send a message.')
    } catch (fetchError) {
      setError(fetchError instanceof Error ? fetchError.message : 'Unknown error')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchRandomInsight()
  }, [])

  return (
    <>
      <div>
        <h1 className="text-red-500">My ML in Financial Mathematics Dashboard</h1>
      </div>
      <div className="card">
        <p>{isLoading ? 'Loading...' : message}</p>
        {error && <p className="text-red-600">Error: {error}</p>}
        <button onClick={fetchRandomInsight} disabled={isLoading}>
          {isLoading ? 'Refreshing...' : 'Get another insight'}
        </button>
      </div>
    </>
  )
}

export default App
