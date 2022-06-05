import {createRouter, createWebHistory} from 'vue-router'
import Home from './components/Home'
import Stocks from './components/Stocks'

const routes = [
    {
        path:'/',
        name:'home',
        component:Home
    },
    {
        path:'/stocks',
        name:'stocks',
        component:Stocks
    }
]

const router = createRouter({
    history:createWebHistory(),
    routes
})

export default router;