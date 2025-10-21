import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import NewsPredictor from '../src/NewsPredictor'
function App() {
  const [count, setCount] = useState(0)

  return (
    <>
     <NewsPredictor />

    </>
  )
}

export default App
