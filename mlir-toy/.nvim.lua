vim.api.nvim_create_autocmd({ "BufNewFile", "BufRead" }, {
	pattern = "*.mlir",
	callback = function()
		vim.api.nvim_set_option_value("filetype", "mlir", { buf = 0 })
	end,
})

vim.opt.tabstop = 2
vim.opt.shiftwidth = 2

require("lspconfig").mlir_lsp_server.setup({})
require("lspconfig").tblgen_lsp_server.setup({})
